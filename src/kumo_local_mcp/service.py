from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .core import Config, Failure, Store, fingerprint, within
from .data import DataManager, read_frame
from .jobs import JobManager
from .models import BackendOptions


class Service:
    def __init__(self, config: Config):
        self.config = config
        self.store = Store(config.output_directory / ".kumo")
        self.data = DataManager(config, self.store)
        self.jobs = JobManager(config, self.store)
        self.capabilities = {}

    async def capability(self, family, refresh=False):
        if family not in self.config.enabled_backends:
            return {
                "family": family,
                "enabled": False,
                "available": False,
                "operations": [],
                "limitations": ["Enable and install this backend using setup."],
            }
        if refresh or family not in self.capabilities:
            response = await self.jobs.worker("probe:" + family, family).call(
                {"operation": "probe", "family": family}
            )
            if response.get("status") == "failed":
                response = {"available": False, "operations": [], "error": response["error"]}
            self.capabilities[family] = {"family": family, "enabled": True, **response}
            # Discovery must not retain a heavyweight import-only process indefinitely.
            worker = self.jobs.worker("probe:" + family, family)
            async with worker.lock:
                await worker.stop()
        return self.capabilities[family]

    async def require(self, family, operation):
        cap = await self.capability(family)
        if not cap["available"]:
            raise Failure(
                "BACKEND_UNAVAILABLE",
                "The requested backend is disabled or unavailable.",
                f"Run kumo-local-mcp setup --backend {family} --install.",
            )
        if operation not in cap["operations"]:
            raise Failure(
                "OPERATION_UNSUPPORTED",
                "The installed adapter does not support this operation.",
                "Read list_capabilities for supported and experimental operations.",
            )

    def attach_dataset(self, payload, object_id, field):
        payload[field] = self.data.descriptor(object_id)["path"]

    def attach_schema(self, payload, schema_id):
        schema = self.store.get(schema_id, "schema")
        findings = self.data.validate_schema(
            schema["tables"], schema["relationships"], schema.get("task_link")
        )
        if findings:
            raise Failure(
                "SCHEMA_INVALID",
                "The relational schema failed validation.",
                "Run validate_relational_schema.",
            )
        schema = {
            **schema,
            "tables": {
                name: {**spec, "path": self.data.descriptor(spec["dataset_id"])["path"]}
                for name, spec in schema["tables"].items()
            },
        }
        payload["schema"] = schema

    def session(self, session_id, family, payload):
        if not session_id:
            return
        session = self.store.get(session_id, "session")
        if session["state"] != "active" or session["family"] != family:
            raise Failure(
                "SESSION_EXPIRED",
                "The session is unavailable or belongs to another model family.",
                "Create a new session after restart or close.",
            )
        for key in ("context_path", "target", "task", "options", "checkpoint"):
            if key in session["settings"] and session["settings"][key] != payload.get(key):
                raise Failure(
                    "SESSION_MISMATCH",
                    "Prediction settings differ from the prepared session.",
                    "Use the original session settings or create a new session.",
                )
        self.store.update(session_id, last_used=time.time())

    def checkpoint(self, payload, checkpoint_id, family):
        if not checkpoint_id:
            return
        checkpoint = self.store.get(checkpoint_id, "checkpoint")
        if checkpoint["family"] != family:
            raise Failure("CHECKPOINT_MISMATCH", "Checkpoint belongs to a different model family.")
        path = within(Path(checkpoint["path"]), [self.store.root])
        settings = checkpoint["settings"]
        if family in {"tabular", "relational"} and (
            settings["task"] != payload["task"] or settings["size"] != payload["options"]["size"]
        ):
            raise Failure(
                "CHECKPOINT_MISMATCH", "Use the task and size recorded with this checkpoint."
            )
        if (
            family == "forecast"
            and settings.get("cross_channel") != payload["options"]["cross_channel"]
        ):
            raise Failure(
                "CHECKPOINT_MISMATCH", "Use the checkpoint's cross_channel model setting."
            )
        payload["checkpoint"] = str(path)

    def validate_tabular(self, payload, evaluation=False, split=None):
        context = read_frame(Path(payload["context_path"]), self.config.max_rows)
        query = read_frame(Path(payload["query_path"]), self.config.max_rows)
        target = payload["target"]
        if target not in context or context[target].isna().any():
            raise Failure("TARGET_INVALID", "Context must contain a complete target column.")
        if payload["task"] == "classification" and context[target].nunique() < 2:
            raise Failure(
                "TARGET_INVALID", "Classification context must contain at least two classes."
            )
        if payload["task"] == "regression" and not pd.api.types.is_numeric_dtype(context[target]):
            raise Failure("TARGET_INVALID", "Regression requires a numeric target.")
        features = context.drop(columns=target)
        query_features = query.drop(columns=target, errors="ignore")
        if list(features.columns) != list(query_features.columns) or not len(features.columns):
            raise Failure(
                "FEATURE_MISMATCH", "Context and query must have the same ordered feature columns."
            )
        if evaluation:
            if target not in query or query[target].isna().any():
                raise Failure("LABELS_REQUIRED", "Evaluation requires complete validation targets.")
            # Includes the target so ordinary repeated feature combinations are allowed.
            a = set(pd.util.hash_pandas_object(context, index=False))
            b = set(pd.util.hash_pandas_object(query[context.columns], index=False))
            if a.intersection(b):
                raise Failure(
                    "SPLIT_OVERLAP",
                    "Context and validation contain overlapping rows.",
                    "Provide disjoint held-out datasets.",
                )
            if payload["task"] == "classification" and not set(query[target]).issubset(
                set(context[target])
            ):
                raise Failure(
                    "UNSEEN_LABEL", "Validation contains classes absent from the context."
                )
            if split and split["method"] == "temporal":
                column = split.get("time_column")
                if not column or column not in context or column not in query:
                    raise Failure(
                        "SPLIT_INVALID",
                        "Temporal evaluation requires a time column in both datasets.",
                    )
                past = pd.to_datetime(context[column], errors="coerce", utc=True)
                future = pd.to_datetime(query[column], errors="coerce", utc=True)
                if past.isna().any() or future.isna().any() or past.max() >= future.min():
                    raise Failure(
                        "SPLIT_INVALID",
                        "All context timestamps must precede validation timestamps.",
                    )

    async def tabular(self, request, family="tabular", operation="predict", split=None):
        await self.require(family, "evaluate" if operation == "evaluate" else operation)
        payload = request.model_dump()
        payload["options"] = payload.pop("backend_options")
        self.attach_dataset(payload, payload.pop("context_dataset_id"), "context_path")
        self.attach_dataset(payload, payload.pop("query_dataset_id"), "query_path")
        if family == "relational":
            if payload.pop("predictive_query", None):
                raise Failure(
                    "PQL_UNSUPPORTED",
                    "Native SDM relational prediction uses labeled task rows and a task link.",
                    "Use Engine tools for predictive queries.",
                )
            self.attach_schema(payload, payload.pop("schema_id"))
            schema = payload["schema"]
            if not schema.get("task_link"):
                raise Failure("SCHEMA_INVALID", "Relational prediction requires one task link.")
            if any(spec.get("time_column") for spec in schema["tables"].values()):
                raise Failure(
                    "TEMPORAL_UNSUPPORTED",
                    "This SDM adapter cannot filter related tables by prediction time.",
                    "Use Engine for point-in-time relational tasks.",
                )
            task_col = schema["task_link"]["task_column"]
            for field in ("context_path", "query_path"):
                task_frame = read_frame(Path(payload[field]), self.config.max_rows)
                linked = read_frame(
                    Path(schema["tables"][schema["task_link"]["table"]]["path"]),
                    self.config.max_rows,
                )
                if (
                    task_col not in task_frame
                    or not task_frame[task_col]
                    .isin(linked[schema["task_link"]["table_column"]])
                    .all()
                ):
                    raise Failure(
                        "SCHEMA_INVALID", "Task rows contain missing or unmatched task-link keys."
                    )
            # Target values must never be present in attached graph tables.
            for spec in schema["tables"].values():
                if payload["target"] in self.data.descriptor(spec["dataset_id"])["columns"]:
                    raise Failure(
                        "TARGET_LEAKAGE",
                        "A related table contains the task target column.",
                        "Register feature-only related tables.",
                    )
        self.validate_tabular(payload, evaluation=operation == "evaluate", split=split)
        self.checkpoint(payload, payload.pop("checkpoint_id", None), family)
        session_id = payload.pop("session_id", None)
        self.session(session_id, family, payload)
        if split:
            payload["evaluation_split"] = split
        return self.jobs.submit(family, operation, payload, session_id)

    def validate_forecast(self, payload, backtest=False):
        df = read_frame(Path(payload["dataset_path"]), self.config.max_rows)
        columns = payload["targets"] + payload["covariates"]
        if any(c not in df or not pd.api.types.is_numeric_dtype(df[c]) for c in columns):
            raise Failure(
                "COLUMN_INVALID",
                "Forecast targets and covariates must be existing numeric columns.",
            )
        if payload["timestamp"] not in df or len(df) < 512:
            raise Failure(
                "HISTORY_INVALID",
                "Forecasting requires a timestamp column and at least 512 historical rows.",
            )
        timestamps = pd.to_datetime(df[payload["timestamp"]], errors="coerce", utc=True)
        if (
            timestamps.isna().any()
            or not timestamps.is_monotonic_increasing
            or timestamps.duplicated().any()
        ):
            raise Failure("TIME_INVALID", "Timestamps must be valid, unique, and increasing.")
        differences = timestamps.diff().dropna()
        if differences.nunique() != 1:
            raise Failure(
                "TIME_IRREGULAR", "Timestamps must be regularly spaced for this forecast adapter."
            )
        if payload["missing_values"] == "reject" and df[columns].isna().any().any():
            raise Failure(
                "MISSING_VALUES", "The selected missing-value policy rejects incomplete series."
            )
        differences = timestamps.diff().dropna()
        if (
            timestamps.isna().any()
            or not timestamps.is_monotonic_increasing
            or timestamps.duplicated().any()
            or differences.nunique() != 1
        ):
            raise Failure(
                "TIMESTAMPS_INVALID",
                "Timestamps must be valid, strictly increasing, and regularly spaced.",
            )
        if payload["missing_values"] == "reject" and df[columns].isna().any().any():
            raise Failure(
                "MISSING_VALUES", "The selected missing-value policy rejects this history."
            )
        if df[columns].isna().all().any():
            raise Failure(
                "MISSING_VALUES", "Entirely missing targets or covariates are not supported."
            )
        if backtest:
            required = (
                payload["initial_history"]
                + (payload["folds"] - 1) * payload["stride"]
                + payload["horizon"]
            )
            if payload["initial_history"] < 512 or required > len(df):
                raise Failure(
                    "SPLIT_INVALID",
                    "Rolling backtest windows exceed the data or use fewer than 512 history rows.",
                )

    async def forecast(self, request, backtest=None):
        await self.require("forecast", "evaluate" if backtest else "predict")
        payload = request.model_dump()
        payload["options"] = payload.pop("backend_options")
        self.checkpoint(payload, payload.pop("checkpoint_id", None), "forecast")
        self.attach_dataset(payload, payload.pop("dataset_id"), "dataset_path")
        if backtest:
            payload.update(backtest)
        self.validate_forecast(payload, bool(backtest))
        session_id = payload.pop("session_id", None)
        self.session(session_id, "forecast", payload)
        return self.jobs.submit(
            "forecast", "backtest" if backtest else "predict", payload, session_id
        )

    async def anomaly(self, request, evaluate=False, label_column=None):
        await self.require("anomaly", "evaluate" if evaluate else "predict")
        payload = request.model_dump()
        payload["options"] = payload.pop("backend_options")
        self.checkpoint(payload, payload.pop("checkpoint_id", None), "anomaly")
        self.attach_dataset(payload, payload.pop("dataset_id"), "dataset_path")
        df = read_frame(Path(payload["dataset_path"]), self.config.max_rows)
        if len(set(payload["columns"])) != len(payload["columns"]) or any(
            c not in df or not pd.api.types.is_numeric_dtype(df[c]) for c in payload["columns"]
        ):
            raise Failure(
                "COLUMN_INVALID", "Anomaly inputs must be distinct existing numeric columns."
            )
        if df[payload["columns"]].isna().all().any():
            raise Failure(
                "MISSING_VALUES", "Entirely missing anomaly input columns are not supported."
            )
        if label_column:
            if label_column in payload["columns"]:
                raise Failure(
                    "TARGET_LEAKAGE", "Ground-truth labels cannot be anomaly input features."
                )
            if (
                label_column not in df
                or df[label_column].isna().any()
                or not set(df[label_column].unique()).issubset({0, 1, False, True})
            ):
                raise Failure(
                    "LABELS_INVALID", "Anomaly evaluation requires complete binary 0/1 labels."
                )
        payload["label_column"] = label_column
        session_id = payload.pop("session_id", None)
        self.session(session_id, "anomaly", payload)
        return self.jobs.submit(
            "anomaly", "evaluate" if evaluate else "predict", payload, session_id
        )

    async def create_session(
        self, family, context_dataset_id=None, target=None, task=None, options=None
    ):
        await self.require(family, "session")
        payload = {
            "options": (options or BackendOptions()).model_dump(),
            "task": task,
            "target": target,
        }
        if family in {"tabular", "relational"}:
            if not context_dataset_id or not target or not task:
                raise Failure(
                    "SESSION_INVALID",
                    "Tabular sessions require a context dataset, target, and task.",
                )
            if family == "relational":
                raise Failure(
                    "OPERATION_UNSUPPORTED",
                    "Reusable relational sessions are not currently exposed.",
                )
            self.attach_dataset(payload, context_dataset_id, "context_path")
            payload["query_path"] = payload["context_path"]
            self.validate_tabular(payload)
        session = self.store.put(
            "session", {"family": family, "state": "active", "settings": payload}
        )
        result = self.jobs.submit(family, "prepare_session", payload, session["id"])
        return {
            **result,
            "session_id": session["id"],
            "warnings": ["Idle unloading releases weights; next use reloads the session."],
        }

    async def finetune(self, request):
        await self.require(request.family, "finetune")
        payload = request.model_dump()
        payload["options"] = payload.pop("backend_options")
        self.attach_dataset(payload, payload.pop("dataset_id"), "dataset_path")
        self.attach_dataset(payload, payload.pop("validation_dataset_id"), "validation_path")
        if request.family in {"forecast", "anomaly"}:
            if payload.pop("session_id", None) or payload.pop("checkpoint_id", None):
                raise Failure(
                    "TRAINING_INVALID",
                    "Start adaptation from the pretrained model without a session or checkpoint.",
                )
            training = read_frame(Path(payload["dataset_path"]), self.config.max_rows)
            validation = read_frame(Path(payload["validation_path"]), self.config.max_rows)
            if fingerprint(Path(payload["dataset_path"])) == fingerprint(
                Path(payload["validation_path"])
            ):
                raise Failure("SPLIT_OVERLAP", "Training and validation datasets must be separate.")
            if request.family == "forecast":
                self.validate_forecast(payload)
                self.validate_forecast({**payload, "dataset_path": payload["validation_path"]})
                if min(len(training), len(validation)) < 512 + payload["horizon"]:
                    raise Failure(
                        "TRAINING_INVALID",
                        "Each forecast split needs at least 512 history rows plus the horizon.",
                    )
                past = pd.to_datetime(training[payload["timestamp"]], utc=True)
                future = pd.to_datetime(validation[payload["timestamp"]], utc=True)
                if past.max() >= future.min():
                    raise Failure(
                        "SPLIT_OVERLAP",
                        "Forecast validation timestamps must follow all training timestamps.",
                    )
            else:
                for frame in (training, validation):
                    if len(frame) < payload["window_rows"] or any(
                        c not in frame
                        or not pd.api.types.is_numeric_dtype(frame[c])
                        or frame[c].isna().any()
                        for c in payload["columns"]
                    ):
                        raise Failure(
                            "TRAINING_INVALID",
                            "Anomaly adaptation requires complete numeric training and validation windows.",
                        )
            return self.jobs.submit(request.family, "finetune", payload)
        payload["context_path"], payload["query_path"] = (
            payload["dataset_path"],
            payload["validation_path"],
        )
        if request.family == "relational":
            if not request.schema_id:
                raise Failure("SCHEMA_INVALID", "Relational fine-tuning requires a schema ID.")
            self.attach_schema(payload, request.schema_id)
            if not payload["schema"].get("task_link"):
                raise Failure("SCHEMA_INVALID", "Relational fine-tuning requires a task link.")
            for spec in payload["schema"]["tables"].values():
                if spec.get("time_column"):
                    raise Failure(
                        "TEMPORAL_UNSUPPORTED", "Use Engine for temporal relational training."
                    )
                if payload["target"] in self.data.descriptor(spec["dataset_id"])["columns"]:
                    raise Failure("TARGET_LEAKAGE", "Related tables must exclude the task target.")
        self.validate_tabular(payload, evaluation=True)
        df = read_frame(Path(payload["dataset_path"]), self.config.max_rows)
        if len(df) < payload["context_rows"] + payload["query_rows"]:
            raise Failure(
                "TRAINING_INVALID",
                "Training requires disjoint context and query batches of the requested sizes.",
            )
        return self.jobs.submit(request.family, "finetune", payload)

    async def explain(self, result_id):
        result = self.store.get(result_id, "result")
        await self.require(result["family"], "explain")
        payload = dict(result["request"])
        if result["operation"] not in {"predict", "evaluate"}:
            raise Failure("OPERATION_UNSUPPORTED", "Explanations require a prediction result.")
        return self.jobs.submit(result["family"], "explain", payload)

    def result_summary(self, result_id):
        result = self.store.get(result_id, "result")
        response = {
            "status": "completed",
            "result_id": result_id,
            "family": result["family"],
            "operation": result["operation"],
            "warnings": result.get("warnings", []),
            "resource_uri": f"kumo://results/{result_id}",
        }
        for key in ("checkpoint_id", "graph_id", "snapshot_id", "training_data_directory"):
            if key in result:
                response[key] = result[key]
        if result.get("output_file"):
            response["output_file"] = result["output_file"]
        if self.config.disclosure.summaries:
            response["summary"] = {
                "rows_processed": result.get("rows_processed", 0),
                "metrics": result.get("metrics", {}),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        if result.get("method"):
            response["method"] = result["method"]
            response["limitations"] = result.get("limitations")
        return response

    def result_path(self, result_id):
        result = self.store.get(result_id, "result")
        if not result.get("output_file"):
            raise Failure("OUTPUT_UNAVAILABLE", "This operation did not produce a tabular output.")
        return within(Path(result["output_file"]), [self.store.root / "results"])

    async def engine(self, operation, schema_id=None, request=None, graph_id=None):
        await self.require("engine", operation)
        payload = request.model_dump() if request else {}
        if operation == "graph":
            payload["schema_id"] = schema_id
        elif operation == "snapshot":
            graph = self.store.get(graph_id, "graph")
            payload["schema_id"] = graph["schema_id"]
        else:
            snapshot = self.store.get(payload.pop("snapshot_id"), "snapshot")
            payload["schema_id"] = snapshot["schema_id"]
        self.attach_schema(payload, payload["schema_id"])
        for relation in payload["schema"]["relationships"]:
            destination = payload["schema"]["tables"][relation["to_table"]]
            if destination.get("primary_key") != relation["to_column"]:
                raise Failure(
                    "SCHEMA_INVALID",
                    "Engine relationships must reference the destination table's primary key.",
                    "Set the destination primary_key or correct the relationship.",
                )
        if operation == "predict":
            self.checkpoint(payload, payload.pop("checkpoint_id"), "engine")
        return self.jobs.submit("engine", operation, payload)

    async def close(self):
        await self.jobs.close()
        self.store.close()
