"""Local Engine adapter. Cloud connectors and arbitrary model plans are not exposed."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

from .core import Config, Failure, within


def execute_engine(req):
    import pandas as pd
    from kumo.engine import (
        Edge,
        FileConnector,
        Graph,
        PredictiveQuery,
        Table,
        Trainer,
        TrainingTableOutputConfig,
        configure,
    )

    config = Config.model_validate(req["config"])
    output = Path(req["output"])
    scratch = config.output_directory / ".kumo" / "engine"
    configure(scratch_dir=str(scratch))
    tables = {}
    schema = req["schema"]
    for name, spec in schema["tables"].items():
        path = Path(spec["path"])
        connector = FileConnector(root_dir=str(path.parent))
        options = {k: spec[k] for k in ("primary_key", "time_column") if spec.get(k)}
        tables[name] = Table.from_source_table(connector[path.stem], **options)
    edges = [
        Edge(src_table=r["from_table"], fkey=r["from_column"], dst_table=r["to_table"])
        for r in schema["relationships"]
    ]
    graph = Graph(tables=tables, edges=edges)
    graph.validate()
    operation = req["operation"]
    if operation == "graph":
        return pd.DataFrame(), {"engine_graph_id": str(graph.id)}
    snapshot = graph.snapshot()
    if operation == "snapshot":
        return pd.DataFrame(), {"engine_snapshot_id": str(snapshot.id)}
    query = PredictiveQuery(graph, req["predictive_query"])
    validation = query.validate()
    # Parse/validation precedes all task data generation and model training.
    if hasattr(validation, "errors") and validation.errors:
        raise Failure("QUERY_INVALID", "The predictive query failed Engine validation.")
    if operation == "generate_training_data":
        generated = query.generate_training_table(snapshot)
        export_dir = output / "training_data"
        generated.export(
            TrainingTableOutputConfig(
                FileConnector(root_dir=str(output)), table_name="training_data/"
            )
        )
        return pd.DataFrame(), {
            "training_data_directory": str(export_dir),
            "warnings": ["Training data retains Engine split assignments."],
        }
    if operation == "train":
        plan = query.suggest_model_plan(snapshot)
        job_plan = getattr(plan, "optimization", None)
        if job_plan is None or not hasattr(job_plan, "max_epochs"):
            raise Failure(
                "ENGINE_API_UNSUPPORTED",
                "Installed Engine does not expose the audited max_epochs setting.",
                "Use a supported Engine version; arbitrary model plans are not accepted.",
            )
        job_plan.max_epochs = req["max_epochs"]
        job_plan.min_steps_per_epoch = 1
        job_plan.max_steps_per_epoch = req["max_steps_per_epoch"]
        job_plan.max_val_steps = req["max_eval_steps"]
        job_plan.max_test_steps = req["max_eval_steps"]
        plan.training_job.num_experiments = req["num_experiments"]
        plan.training_job.refit_trainval = req["refit_trainval"]
        plan.training_job.refit_full = False
        kwargs = {}
        if "accelerator" in inspect.signature(Trainer).parameters:
            kwargs["accelerator"] = {"cuda": "gpu", "cpu": "cpu", "mps": "mps", "auto": "auto"}[
                config.device
            ]
        elif config.device != "auto":
            raise Failure(
                "DEVICE_UNSUPPORTED",
                "Installed Engine cannot accept the selected device explicitly.",
            )
        trainer = Trainer(plan, **kwargs)
        trained = trainer.fit(query.generate_training_table(snapshot), snapshot=snapshot)
        destination = output / "model_bundle"
        source = within(Path(trained.bundle_dir), [scratch])
        shutil.copytree(source, destination)
        return pd.DataFrame(), {
            "checkpoint": str(destination),
            "metrics": dict(trained.metrics),
            "model_settings": {"query": req["predictive_query"]},
        }
    if operation == "predict":
        checkpoint = within(Path(req["checkpoint"]), [config.output_directory / ".kumo"])
        kwargs = {}
        if "accelerator" in inspect.signature(Trainer.load).parameters:
            kwargs["accelerator"] = {"cuda": "gpu", "cpu": "cpu", "mps": "mps", "auto": "auto"}[
                config.device
            ]
        trainer = Trainer.load(str(checkpoint), **kwargs)
        result = trainer.predict(
            query.generate_prediction_table(snapshot),
            snapshot=snapshot,
            binary_classification_threshold=req["threshold"],
        )
        if result.num_rows > config.max_rows:
            raise Failure("DATA_LIMIT", "Engine output exceeds the configured in-memory row limit.")
        return result.frame, {}
    raise Failure("OPERATION_UNSUPPORTED", "This Engine operation is not implemented.")
