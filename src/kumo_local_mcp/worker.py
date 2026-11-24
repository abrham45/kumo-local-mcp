"""Private line-JSON worker. No assistant-supplied code, imports, or shell commands."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import sys
import time
from pathlib import Path

from .core import Config, Failure, atomic_json
from .data import read_frame

CLASSES = {
    "tabular": "KumoTabular",
    "relational": "KumoRelational",
    "forecast": "KumoForecast",
    "anomaly": "KumoAnomaly",
}
MODELS = {}
FITTED = {}


def probe(family):
    """Import, but never instantiate or download model weights."""
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if family == "engine":
        from kumo.engine import Graph, PredictiveQuery, Trainer

        ok = all(
            hasattr(cls, member)
            for cls, member in [
                (Graph, "snapshot"),
                (Trainer, "fit"),
                (Trainer, "predict"),
                (Trainer, "load"),
                (PredictiveQuery, "generate_training_table"),
            ]
        )
        return {
            "status": "completed",
            "available": ok,
            "version": importlib.metadata.version("kumo-engine"),
            "devices": devices,
            "operations": ["graph", "snapshot", "generate_training_data", "train", "predict"]
            if ok
            else [],
            "experimental": ["engine_adapter"],
            "limitations": ["Local files only; no cloud connectors."],
        }
    import sdm

    available = hasattr(sdm.models, CLASSES[family])
    hardware_devices = devices.copy()
    devices = [device for device in devices if device != "mps"]
    operations = ["predict", "evaluate", "session"] if available else []
    if family in {"tabular", "relational"} and available:
        import sdm.explain

        if hasattr(sdm.explain, "GradientExplainer"):
            operations.append("explain")
    if family == "relational" and "session" in operations:
        operations.remove("session")
    if available:
        operations.append("finetune")
    return {
        "status": "completed",
        "available": available,
        "version": importlib.metadata.version("structured-data-models"),
        "devices": devices,
        "hardware_devices": hardware_devices,
        "model_class": CLASSES[family],
        "operations": operations,
        "experimental": (
            ["relational_adapter"]
            if family == "relational"
            else ["timeseries_finetuning"]
            if family in {"forecast", "anomaly"}
            else []
        ),
        "limitations": {
            "tabular": ["One target; fine-tuning requires separate validation data."],
            "relational": [
                "Native SDM task rows and keys; PQL requires Engine.",
                "Time columns are metadata; temporal graph filtering is not implemented.",
                "Relational fine-tuning is experimental.",
            ],
            "forecast": [
                "At least 512 regularly spaced observations; past-only covariates.",
                "Point forecasts; default zero imputation.",
                "Fine-tuning is experimental and requires a later validation period.",
            ],
            "anomaly": [
                "Default mean imputation; MAE scores and SCS/MACS labels.",
                "Gradient explanations are unavailable; experimental adaptation assumes nominal data.",
            ],
        }[family],
    }


def device_for(config):
    import torch

    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if config.device == "cuda" and not torch.cuda.is_available():
        raise Failure(
            "DEVICE_UNAVAILABLE",
            "CUDA is not available in this worker.",
            "Use CPU or install CUDA locally.",
        )
    if config.device == "mps":
        if not torch.backends.mps.is_available():
            raise Failure("DEVICE_UNAVAILABLE", "MPS is not available in this worker.")
        raise Failure(
            "DEVICE_UNSUPPORTED",
            "SDM adapters currently validate CPU and CUDA execution only.",
            "Choose cpu or auto. Engine can select MPS when its installed API supports it.",
        )
    return config.device


def model_for(req):
    import sdm
    import torch

    family, options = req["family"], req["options"]
    config = Config.model_validate(req["config"])
    device = device_for(config)
    settings = {"device": device}
    if family in {"tabular", "relational"}:
        settings["task"] = req["task"]
    if family == "tabular":
        settings["size"] = options["size"]
    if family == "forecast":
        settings["cross_channel"] = options["cross_channel"]
    checkpoint = req.get("checkpoint")
    settings["pretrained"] = not bool(checkpoint)
    key = json.dumps([family, settings, checkpoint], sort_keys=True)
    if key not in MODELS:
        # Bound per-worker model residency. Persistent sessions retain only their active model.
        MODELS.clear()
        FITTED.clear()
        cls = getattr(sdm.models, CLASSES[family], None)
        if cls is None:
            raise Failure(
                "BACKEND_UNSUPPORTED",
                "The installed SDM version lacks this model family.",
                "Run setup to install the supported SDM release.",
            )
        model = cls(**settings)
        if checkpoint:
            from safetensors.torch import load_file

            model.load_state_dict(load_file(checkpoint, device=device))
        model.eval()
        MODELS[key] = model
    generator = torch.Generator(device=device).manual_seed(options["seed"])
    return MODELS[key], device, generator


def frame(req, name):
    return read_frame(Path(req[name]), req["config"]["max_rows"])


def tensor(df, device, target=None, task=None, ids=(), times=(), numerical=()):
    import sdm

    overrides = {column: "id" for column in ids}
    overrides.update({column: "numerical" for column in numerical})
    overrides.update({column: "datetime" for column in times})
    if target:
        overrides[target] = "categorical" if task == "classification" else "numerical"
    return sdm.TableTensor.from_pandas(
        df, stypes=sdm.infer_stypes(df, overrides=overrides), device=device
    )


def related(req, device):
    import sdm

    schema = req.get("schema")
    if not schema:
        return None
    keys = {name: set() for name in schema["tables"]}
    for rel in schema["relationships"]:
        keys[rel["from_table"]].add(rel["from_column"])
        keys[rel["to_table"]].add(rel["to_column"])
    link = schema["task_link"]
    keys[link["table"]].add(link["table_column"])
    tables = {
        name: tensor(
            read_frame(Path(spec["path"]), req["config"]["max_rows"]), device, ids=keys[name]
        )
        for name, spec in schema["tables"].items()
    }
    return sdm.RelatedTables(
        tables=tables,
        relationships=[
            {
                "left_table": r["from_table"],
                "left_column": r["from_column"],
                "right_table": r["to_table"],
                "right_column": r["to_column"],
            }
            for r in schema["relationships"]
        ],
        task_links=[link],
    )


def tabular_inputs(req, device):
    context, query = frame(req, "context_path"), frame(req, "query_path")
    target, task = req["target"], req["task"]
    ids = [req["schema"]["task_link"]["task_column"]] if req.get("schema") else []
    x = tensor(context.drop(columns=target), device, ids=ids)
    y = tensor(context[[target]], device, target, task)
    q = tensor(query.drop(columns=[target], errors="ignore"), device, ids=ids)
    return x, y, q


def supervised_prediction(req, explain=False):
    model, device, generator = model_for(req)
    x, y, query = tabular_inputs(req, device)
    graph = related(req, device)
    kwargs = {
        "x_context": x,
        "y_context": y,
        "x_query": query,
        "num_estimators": req["options"]["num_estimators"],
        "generator": generator,
    }
    if graph is not None:
        kwargs.update(related_context_tables=graph, related_query_tables=graph)
    if explain:
        from sdm.explain import GradientExplainer

        kwargs.pop("x_query")
        kwargs["num_estimators"] = 1
        attribution, related_attribution = GradientExplainer().explain(model, query, **kwargs)
        import pandas as pd

        # Include graph-feature gradients: relational task rows often contain only IDs.
        # Positional coordinates keep schema names out of returned sample values.
        records = []
        blocks = [("task", 0, attribution)]
        if related_attribution is not None:
            blocks.extend(
                ("related", index, block)
                for index, block in enumerate(related_attribution.tables.values(), start=1)
            )
        for scope, table_index, block in blocks:
            values = block.numerical.detach().abs().mean(dim=0).cpu().numpy()
            for row_index in range(values.shape[-2]):
                for feature_index in range(values.shape[-1]):
                    records.append(
                        (
                            scope,
                            table_index,
                            row_index,
                            feature_index,
                            float(values[row_index, feature_index]),
                        )
                    )
        return pd.DataFrame(
            records,
            columns=["scope", "table_index", "row_index", "feature_index", "absolute_gradient"],
        ), {
            "method": "absolute_gradient",
            "limitations": "Sensitivity of preprocessed numerical inputs, averaged over outputs; positional coordinates; not causal.",
        }
    if req.get("session_id") and graph is None:
        session_key = json.dumps(
            [req["context_path"], req["target"], req["task"], req["options"]], sort_keys=True
        )
        if FITTED.get(req["session_id"]) != session_key:
            model.fit(
                x=x, y=y, num_estimators=req["options"]["num_estimators"], generator=generator
            )
            FITTED[req["session_id"]] = session_key
        prediction = model.predict(query)
    else:
        prediction = model(**kwargs)
    return prediction.to_pandas(), {}


def forecast_prediction(req, df=None, explain=False):
    import pandas as pd
    import sdm

    model, device, generator = model_for(req)
    df = frame(req, "dataset_path") if df is None else df
    cols = req["targets"] + req["covariates"]
    values = df[cols].copy()
    if req["missing_values"] == "interpolate":
        # Uses history only. Backtests call this independently at each cutoff.
        values = values.interpolate().ffill().bfill()
    x = tensor(values[req["covariates"]], device, numerical=req["covariates"])
    y = tensor(values[req["targets"]], device, numerical=req["targets"])
    query = sdm.TableTensor(size=(req["horizon"],), device=device)
    kwargs = dict(
        x_context=x,
        y_context=y,
        num_estimators=req["options"]["num_estimators"],
        generator=generator,
        alpha=req["options"]["alpha"],
        k=req["options"]["k"],
        temperature=req["options"]["temperature"],
    )
    if explain:
        raise Failure(
            "OPERATION_UNSUPPORTED",
            "Forecast gradient explanations require query features; this adapter uses past-only covariates.",
            "Use backtest_forecast to inspect held-out forecast errors.",
        )
    out = model(x_query=query, **kwargs).to_pandas()
    stamps = pd.to_datetime(df[req["timestamp"]], utc=True)
    frequency = stamps.iloc[-1] - stamps.iloc[-2]
    out.insert(
        0,
        req["timestamp"],
        pd.date_range(stamps.iloc[-1] + frequency, periods=req["horizon"], freq=frequency),
    )
    return out, {"missing_values": req["missing_values"], "historical_rows": len(df)}


def anomaly_prediction(req):
    model, device, generator = model_for(req)
    values = frame(req, "dataset_path")[req["columns"]]
    out = model(
        tensor(values, device, numerical=req["columns"]),
        generator=generator,
        num_samples=req["options"]["num_samples"],
        num_steps=req["options"]["num_steps"],
    ).to_pandas()
    out["anomaly"] = (
        out[req["label_method"]].astype(bool)
        if req["threshold"] is None
        else out["mae"] >= req["threshold"]
    )
    return out, {"threshold": req["threshold"], "label_method": req["label_method"]}


def supervised_metrics(req, prediction):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
    )

    truth = frame(req, "query_path")[req["target"]]
    if req["task"] == "classification":
        labels = [str(v) for v in prediction.columns]
        y = truth.astype(str)
        if not set(y).issubset(labels):
            raise Failure(
                "UNSEEN_LABEL", "Validation labels include a class absent from the context."
            )
        values = prediction.to_numpy(dtype=float)
        totals = values.sum(axis=1, keepdims=True)
        if not np.isfinite(values).all() or (values < 0).any() or (totals <= 0).any():
            raise Failure(
                "OUTPUT_INVALID", "The backend returned invalid classification probabilities."
            )
        # SDM probabilities originate in float32; normalize rounding before float64 log-loss scoring.
        values = values / totals
        guessed = np.asarray(labels)[values.argmax(axis=1)]
        metrics = {
            "accuracy": float(accuracy_score(y, guessed)),
            "macro_f1": float(f1_score(y, guessed, average="macro", zero_division=0)),
        }
        if len(labels) > 1:
            # Explicit order: sklearn otherwise sorts labels and may misalign probability columns.
            order = np.argsort(labels)
            metrics["log_loss"] = float(
                log_loss(y, values[:, order], labels=np.asarray(labels)[order])
            )
        return metrics
    estimate = prediction["q500"].to_numpy()
    return {
        "mae": float(mean_absolute_error(truth, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
    }


def backtest(req):
    import numpy as np
    import pandas as pd

    df, folds = frame(req, "dataset_path"), []
    for index in range(req["folds"]):
        cutoff = req["initial_history"] + index * req["stride"]
        pred, _ = forecast_prediction(req, df.iloc[:cutoff])
        actual = df.iloc[cutoff : cutoff + req["horizon"]][req["targets"]].to_numpy(dtype=float)
        values = pred.drop(columns=req["timestamp"]).to_numpy(dtype=float)
        if values.shape != actual.shape:
            raise Failure("OUTPUT_INVALID", "Forecast output does not match the requested targets.")
        valid = np.isfinite(actual) & np.isfinite(values)
        if not valid.any():
            raise Failure(
                "EVALUATION_INVALID", "A backtest fold has no observed validation targets."
            )
        delta = actual[valid] - values[valid]
        folds.append(
            {
                "fold": index + 1,
                "cutoff": cutoff,
                "observations": int(valid.sum()),
                "mae": float(np.abs(delta).mean()),
                "rmse": float(np.sqrt((delta**2).mean())),
            }
        )
    return pd.DataFrame(folds), {
        "mae_mean_over_folds": float(np.mean([v["mae"] for v in folds])),
        "folds": len(folds),
        "split": "rolling_origin",
    }


def finetune(req):
    import torch
    import sdm
    import sdm.processing as sp
    from safetensors.torch import save_file

    if req["family"] not in {"tabular", "relational"}:
        raise Failure(
            "OPERATION_UNSUPPORTED",
            "Fine-tuning is currently implemented for Tabular only.",
            "Use Engine training for relational models. Other adapters report this limitation.",
        )
    model, device, generator = model_for(req)
    train = frame(req, "dataset_path")
    ids = [req["schema"]["task_link"]["task_column"]] if req.get("schema") else []
    table = tensor(train, device, req["target"], req["task"], ids=ids)
    graph = related(req, device)
    recipe = model.default_recipe()
    recipe.output = sp.Identity()
    if req["task"] == "regression":
        recipe.target = sp.StypeDispatch(numerical=sp.Standardize())
    optimizer = torch.optim.AdamW(model.parameters(), lr=req["learning_rate"])
    evaluation_req = {
        **req,
        "context_path": req["dataset_path"],
        "query_path": req["validation_path"],
        "session_id": None,
    }
    # Select a checkpoint against held-out validation, including the pretrained baseline.
    model.eval()
    baseline, _ = supervised_prediction(evaluation_req)
    best_metrics = supervised_metrics(evaluation_req, baseline)
    metric = "log_loss" if req["task"] == "classification" else "mae"
    if metric not in best_metrics:
        metric = "accuracy"
    best_score = best_metrics[metric]
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    for step in range(req["steps"]):
        perm = torch.randperm(len(table), device=device, generator=generator)
        context = table[perm[: req["context_rows"]]]
        query = table[perm[req["context_rows"] : req["context_rows"] + req["query_rows"]]]
        if req["task"] == "classification":
            # Context contains every training class, while query rows remain disjoint.
            raw = train.iloc[perm.detach().cpu().numpy()]
            if not set(
                raw.iloc[req["context_rows"] : req["context_rows"] + req["query_rows"]][
                    req["target"]
                ]
            ).issubset(set(raw.iloc[: req["context_rows"]][req["target"]])):
                continue
        model.train()
        relation_options = (
            {"related_context_tables": graph, "related_query_tables": graph}
            if graph is not None
            else {}
        )
        out = model(
            x_context=context.drop_columns(req["target"]),
            y_context=context[req["target"]],
            x_query=query.drop_columns(req["target"]),
            recipe=recipe,
            num_estimators=1,
            generator=generator,
            **relation_options,
        )[0]
        if req["task"] == "classification":
            logits, y = sdm.evaluation.to_class_indices(out, query[req["target"]])
            loss = torch.nn.functional.cross_entropy(logits, y)
        else:
            diff = query[req["target"]].numerical - out.numerical
            levels = torch.linspace(0.001, 0.999, 999, device=device)
            loss = torch.maximum(levels * diff, (levels - 1) * diff).mean()
        if not torch.isfinite(loss):
            raise Failure(
                "TRAINING_DIVERGED",
                "Fine-tuning produced a non-finite loss.",
                "Reduce the learning rate.",
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        pred, _ = supervised_prediction(evaluation_req)
        metrics = supervised_metrics(evaluation_req, pred)
        score = metrics[metric]
        improved = score > best_score if metric == "accuracy" else score < best_score
        if improved:
            best_score, best_metrics = score, metrics
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {"step": step + 1, "loss": float(loss.detach().cpu()), "validation_score": score}
        )
    if not history:
        raise Failure(
            "TRAINING_INVALID",
            "No valid disjoint context/query batches could be sampled.",
            "Increase context_rows or supply a better balanced dataset.",
        )
    model.load_state_dict(best)
    model.eval()
    checkpoint = Path(req["output"]) / "model.safetensors"
    save_file({k: v.contiguous() for k, v in best.items()}, str(checkpoint))
    import pandas as pd

    return pd.DataFrame(history), {
        "metrics": best_metrics,
        "checkpoint": str(checkpoint),
        "model_settings": {"task": req["task"], "size": req["options"]["size"]},
        "warnings": ["Checkpoint selected on validation data; reserve a separate test dataset."],
    }


def finetune_timeseries(req):
    import pandas as pd
    import sdm
    import torch
    from safetensors.torch import save_file

    model, device, _ = model_for(req)
    training, validation = frame(req, "dataset_path"), frame(req, "validation_path")
    family = req["family"]
    if family == "forecast" and req.get("trainable_scope", "head") == "head":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("model.head.", "model.cross_channel.")))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise Failure(
            "OPERATION_UNSUPPORTED",
            "The installed model exposes no parameters for this adaptation scope.",
        )
    optimizer = torch.optim.AdamW(trainable, lr=req["learning_rate"])
    history = []

    def loss_for(data, training_step, step):
        generator = torch.Generator(device=device).manual_seed(req["options"]["seed"] + step)
        if family == "forecast":
            length = 512 + req["horizon"]
            start = (step * 17) % (len(data) - length + 1) if training_step else len(data) - length
            window = data.iloc[start : start + length]
            past, truth = window.iloc[:512], window.iloc[512:]
            # Interpolation is fitted to past observations only; future values remain labels.
            columns = req["targets"] + req["covariates"]
            values = past[columns].copy()
            if req["missing_values"] == "interpolate":
                values = values.interpolate().ffill().bfill()
            x = tensor(values[req["covariates"]], device, numerical=req["covariates"])
            y = tensor(values[req["targets"]], device, numerical=req["targets"])
            forecast = model(
                x_context=x,
                y_context=y,
                x_query=sdm.TableTensor(size=(req["horizon"],), device=device),
                num_estimators=1,
                generator=generator,
                alpha=req["options"]["alpha"],
                k=req["options"]["k"],
                temperature=req["options"]["temperature"],
            )
            actual = tensor(truth[req["targets"]], device, numerical=req["targets"]).numerical
            observed = torch.isfinite(actual)
            if not observed.any():
                raise Failure(
                    "TRAINING_INVALID", "A forecast training window has no observed query labels."
                )
            return ((forecast.numerical - actual)[observed] ** 2).mean()
        length = req["window_rows"]
        start = (step * 17) % (len(data) - length + 1) if training_step else len(data) - length
        values = tensor(
            data.iloc[start : start + length][req["columns"]], device, numerical=req["columns"]
        )
        out = model(
            values,
            num_samples=req["options"]["num_samples"],
            num_steps=req["options"]["num_steps"],
            generator=generator,
        )
        return out["mae"].numerical.mean()

    model.eval()
    baseline = float(loss_for(validation, False, 0).detach().cpu())
    best_score = baseline
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for step in range(req["steps"]):
        model.train()
        loss = loss_for(training, True, step + 1)
        if not loss.requires_grad:
            raise Failure(
                "OPERATION_UNSUPPORTED",
                "Installed backend does not preserve gradients for this adaptation objective.",
            )
        if not torch.isfinite(loss):
            raise Failure(
                "TRAINING_DIVERGED",
                "Adaptation produced a non-finite loss.",
                "Reduce the learning rate.",
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        score = float(loss_for(validation, False, 0).detach().cpu())
        if score < best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {"step": step + 1, "loss": float(loss.detach().cpu()), "validation_loss": score}
        )
    model.load_state_dict(best)
    checkpoint = Path(req["output"]) / "model.safetensors"
    save_file({k: v.contiguous() for k, v in best.items()}, str(checkpoint))
    metric = "validation_forecast_mse" if family == "forecast" else "validation_reconstruction_mae"
    warnings = [
        "Experimental adaptation; checkpoint selected on a fixed held-out validation window.",
        "Reserve a separate test period to measure predictive performance.",
    ]
    if family == "forecast" and req.get("trainable_scope", "head") == "head":
        warnings.append(
            "Adapted forecast head and cross-channel layers; pretrained encoder stayed frozen."
        )
    if family == "anomaly":
        warnings.append(
            "Anomaly adaptation assumes nominal training/validation windows; loss is not anomaly accuracy."
        )
    return pd.DataFrame(history), {
        "checkpoint": str(checkpoint),
        "model_settings": {"cross_channel": req["options"]["cross_channel"]}
        if family == "forecast"
        else {},
        "metrics": {metric: best_score, "pretrained_validation_loss": baseline},
        "warnings": warnings,
    }


def execute(req):
    operation = req["operation"]
    if operation == "probe":
        return probe(req["family"])
    if operation == "prepare_session":
        model, device, generator = model_for(req)
        if req["family"] == "tabular":
            context = tensor(frame(req, "context_path"), device, req["target"], req["task"])
            model.fit(
                x=context.drop_columns(req["target"]),
                y=context[req["target"]],
                num_estimators=req["options"]["num_estimators"],
                generator=generator,
            )
            FITTED[req["session_id"]] = json.dumps(
                [req["context_path"], req["target"], req["task"], req["options"]], sort_keys=True
            )
        return {"status": "completed", "warnings": [], "rows_processed": 0}
    start = time.monotonic()
    metrics, extra = {}, {}
    if req["family"] == "engine":
        from .engine import execute_engine

        output, extra = execute_engine(req)
    elif operation == "finetune":
        try:
            output, extra = (
                finetune_timeseries(req)
                if req["family"] in {"forecast", "anomaly"}
                else finetune(req)
            )
            metrics = extra.pop("metrics", {})
        finally:
            # Training must never mutate a future request for the pretrained baseline.
            MODELS.clear()
            FITTED.clear()
    elif req["family"] in {"tabular", "relational"}:
        output, extra = supervised_prediction(req, explain=operation == "explain")
        if operation == "evaluate":
            metrics = supervised_metrics(req, output)
    elif req["family"] == "forecast":
        if operation == "backtest":
            output, metrics = backtest(req)
        else:
            output, extra = forecast_prediction(req, explain=operation == "explain")
    else:
        output, extra = anomaly_prediction(req)
        if operation == "evaluate":
            if req.get("label_column"):
                from sklearn.metrics import precision_score, recall_score, f1_score

                truth = frame(req, "dataset_path")[req["label_column"]].astype(int)
                guess = output["anomaly"].astype(int)
                metrics = {
                    "precision": float(precision_score(truth, guess, zero_division=0)),
                    "recall": float(recall_score(truth, guess, zero_division=0)),
                    "f1": float(f1_score(truth, guess, zero_division=0)),
                }
            else:
                metrics = {
                    "anomalies_flagged": int(output["anomaly"].sum()),
                    "mae_score_mean": float(output["mae"].mean()),
                }
                extra["warnings"] = [
                    "No ground-truth labels supplied; these are descriptive diagnostics, not accuracy."
                ]
    path = Path(req["output"]) / "output.csv"
    if len(output.columns):
        temporary = path.with_suffix(".tmp")
        output.to_csv(temporary, index=False)
        temporary.chmod(0o600)
        temporary.replace(path)
    response = {
        "status": "completed",
        "rows_processed": len(output),
        "metrics": metrics,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "warnings": [],
        **extra,
    }
    if len(output.columns):
        response["output_file"] = str(path)
    atomic_json(Path(req["output"]) / "report.json", response)
    return response


def main():
    # Redirect at OS descriptor level, so native-library stdout cannot corrupt line JSON.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            response = execute(req)
        except Failure as exc:
            response = exc.response()
        except (ImportError, ModuleNotFoundError):
            logging.exception("Backend import failed")
            response = Failure(
                "BACKEND_MISSING",
                "A required backend dependency is missing or incompatible.",
                "Run setup --install for this backend.",
            ).response()
        except Exception:
            logging.exception("Backend operation failed")
            response = Failure(
                "BACKEND_FAILED",
                "The backend operation failed.",
                "Inspect local stderr logs; check weights, data, hardware, and backend version.",
            ).response()
        protocol.write(json.dumps(response, allow_nan=False, default=str) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
