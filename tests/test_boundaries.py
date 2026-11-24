import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from kumo_local_mcp.core import Config, Failure
from kumo_local_mcp.models import BackendOptions
from kumo_local_mcp.service import Service


@pytest.fixture
def service(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    instance = Service(
        Config(allowed_input_directories=[inputs], output_directory=tmp_path / "out")
    )
    yield instance
    asyncio.run(instance.close())


def register(service, name="data.csv", content="private_column,target\n101,0\n102,1\n"):
    path = service.config.allowed_input_directories[0] / name
    path.write_text(content)
    return service.data.register(str(path))


def test_snapshot_is_immutable_and_versioned(service):
    first = register(service)
    second = register(service, content="private_column,target\n103,0\n104,1\n")
    assert first["id"] != second["id"]
    assert first["sha256"] != second["sha256"]
    assert service.data.frame(first["id"]).iloc[0, 0] == 101


def test_symlink_and_prefix_sibling_do_not_bypass_input_roots(service, tmp_path):
    outside = tmp_path / "inputs-other"
    outside.mkdir()
    secret = outside / "secret.csv"
    secret.write_text("secret\n1\n")
    link = service.config.allowed_input_directories[0] / "link.csv"
    link.symlink_to(secret)
    for path in (secret, link):
        with pytest.raises(Failure, match="outside"):
            service.data.register(str(path))


def test_tampered_snapshot_is_rejected(service):
    item = register(service)
    Path(item["path"]).write_text("private_column,target\n999,0\n")
    with pytest.raises(Failure) as exc:
        service.data.descriptor(item["id"])
    assert exc.value.code == "DATA_CHANGED"


def test_disclosure_defaults_hide_names_summaries_and_values(service):
    item = register(service)
    response = service.data.inspect(item["id"])
    assert "private_column" not in json.dumps(response)
    assert "summary" not in response
    with pytest.raises(Failure) as exc:
        service.data.preview(service.data.frame(item["id"]), 10)
    assert exc.value.code == "DISCLOSURE_DENIED"


def test_sample_limit_and_column_anonymization(service):
    item = register(service)
    service.config.disclosure.sample_values = True
    service.config.max_preview_rows = 1
    response = service.data.preview(service.data.frame(item["id"]), 100)
    assert len(response["rows"]) == 1
    assert list(response["rows"][0]) == ["column_1", "column_2"]


def test_result_resource_view_does_not_disclose_raw_request(service):
    result = service.store.put(
        "result",
        {
            "family": "tabular",
            "operation": "predict",
            "request": {"target": "private_column", "database_password": "DO-NOT-RETURN"},
            "metrics": {"accuracy": 0.9},
            "warnings": [],
            "rows_processed": 2,
        },
    )
    rendered = json.dumps(service.result_summary(result["id"]))
    assert "private_column" not in rendered and "DO-NOT-RETURN" not in rendered
    assert "accuracy" not in rendered


def test_row_and_file_size_limits(service):
    service.config.max_rows = 1
    with pytest.raises(Failure) as exc:
        register(service)
    assert exc.value.code == "DATA_LIMIT"


def test_duplicate_csv_headers_are_not_silently_renamed(service):
    with pytest.raises(Failure) as exc:
        register(service, content="secret,secret\n1,2\n")
    assert exc.value.code == "DATA_INVALID"
    service.config.max_rows = 100
    service.config.max_input_bytes = 1
    with pytest.raises(Failure) as exc:
        register(service)
    assert exc.value.code == "DATA_LIMIT"


def test_relational_orphans_and_duplicate_keys_are_reported(service):
    users = register(service, "users.csv", "id,age\n1,20\n2,30\n")
    orders = register(service, "orders.csv", "user_id,value\n1,4\n9,6\n")
    tables = {
        "users": {"dataset_id": users["id"], "primary_key": "id"},
        "orders": {"dataset_id": orders["id"]},
    }
    relations = [
        {"from_table": "orders", "from_column": "user_id", "to_table": "users", "to_column": "id"}
    ]
    assert "orphan" in " ".join(service.data.validate_schema(tables, relations))


def test_engine_cannot_silently_redirect_foreign_key_to_other_column(service, monkeypatch):
    users = register(service, "users.csv", "id,code\n1,11\n2,22\n")
    orders = register(service, "orders.csv", "user_code,value\n11,4\n22,6\n")
    schema = service.store.put(
        "schema",
        {
            "tables": {
                "users": {"dataset_id": users["id"], "primary_key": "id"},
                "orders": {"dataset_id": orders["id"]},
            },
            "relationships": [
                {
                    "from_table": "orders",
                    "from_column": "user_code",
                    "to_table": "users",
                    "to_column": "code",
                }
            ],
        },
    )

    async def available(*args):
        pass

    monkeypatch.setattr(service, "require", available)

    async def run():
        with pytest.raises(Failure) as exc:
            await service.engine("graph", schema_id=schema["id"])
        assert exc.value.code == "SCHEMA_INVALID"
        assert not service.store.list("job")

    asyncio.run(run())


def test_evaluation_rejects_overlap_before_worker_start(service):
    item = register(service)
    payload = {
        "context_path": item["path"],
        "query_path": item["path"],
        "target": "target",
        "task": "classification",
    }
    with pytest.raises(Failure) as exc:
        service.validate_tabular(payload, evaluation=True)
    assert exc.value.code == "SPLIT_OVERLAP"


def test_forecast_rejects_irregular_time_and_excess_backtest_windows(service):
    df = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=520), "target": range(520)})
    path = service.config.allowed_input_directories[0] / "history.csv"
    df.to_csv(path, index=False)
    item = service.data.register(str(path))
    payload = {
        "dataset_path": item["path"],
        "timestamp": "time",
        "targets": ["target"],
        "covariates": [],
        "horizon": 10,
        "missing_values": "backend",
        "initial_history": 512,
        "folds": 2,
        "stride": 10,
    }
    with pytest.raises(Failure) as exc:
        service.validate_forecast(payload, backtest=True)
    assert exc.value.code == "SPLIT_INVALID"
    df.loc[100, "time"] = df.loc[99, "time"]
    df.to_csv(path, index=False)
    item = service.data.register(str(path))
    payload["dataset_path"] = item["path"]
    with pytest.raises(Failure):
        service.validate_forecast(payload)


def test_advanced_options_cannot_inject_code_or_paths():
    for values in ({"shell": "echo bad"}, {"model_path": "/private/secret"}, {"num_estimators": 0}):
        with pytest.raises(ValidationError):
            BackendOptions(**values)


def test_export_rejects_traversal_symlink_and_overwrite(service, tmp_path):
    item = register(service)
    source = Path(item["path"])
    for name in ("../outside.csv", str(tmp_path / "elsewhere.csv")):
        with pytest.raises(Failure):
            service.data.export(source, name, "csv")
    target = service.data.export(source, "safe.csv", "csv")
    assert Path(target["path"]).exists()
    with pytest.raises(Failure):
        service.data.export(source, "safe.csv", "csv")
    (service.config.output_directory / "link").symlink_to(tmp_path)
    with pytest.raises(Failure):
        service.data.export(source, "link/escaped.csv", "csv")


def test_registry_restart_preserves_results_and_interrupts_jobs(tmp_path):
    config = Config(allowed_input_directories=[tmp_path], output_directory=tmp_path / "out")
    first = Service(config)
    result = first.store.put(
        "result", {"family": "tabular", "operation": "predict", "warnings": []}
    )
    job = first.store.put("job", {"state": "running", "progress": 0.1})
    session = first.store.put("session", {"state": "active"})
    asyncio.run(first.close())
    second = Service(config)
    assert second.store.get(result["id"])["family"] == "tabular"
    assert second.store.get(job["id"])["state"] == "interrupted"
    assert second.store.get(session["id"])["state"] == "expired"
    asyncio.run(second.close())


def test_disabled_backend_has_actionable_failure(service):
    async def run():
        with pytest.raises(Failure) as exc:
            await service.require("tabular", "predict")
        assert exc.value.code == "BACKEND_UNAVAILABLE"

    asyncio.run(run())


def test_backend_interpreter_preserves_virtualenv_symlink(tmp_path):
    import sys

    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    config = Config(
        allowed_input_directories=[tmp_path],
        output_directory=tmp_path / "out",
        backend_python={"tabular": interpreter},
    )
    assert config.backend_python["tabular"] == interpreter
    assert config.backend_python["tabular"] != interpreter.resolve()


def test_training_failure_cannot_poison_pretrained_model_cache(monkeypatch):
    from kumo_local_mcp import worker

    worker.MODELS["baseline"] = object()
    worker.FITTED["session"] = "context"

    def fail(request):
        raise Failure("TRAINING_DIVERGED", "test training failure")

    monkeypatch.setattr(worker, "finetune", fail)
    with pytest.raises(Failure):
        worker.execute({"operation": "finetune", "family": "tabular"})
    assert not worker.MODELS and not worker.FITTED


def test_forecast_adaptation_rejects_overlapping_periods(service, monkeypatch):
    from kumo_local_mcp.models import ForecastFineTuneRequest

    async def available(*args):
        pass

    monkeypatch.setattr(service, "require", available)
    ids = []
    for name, start in [("past", "2025-01-01"), ("future", "2025-02-01")]:
        data = pd.DataFrame({"time": pd.date_range(start, periods=520), "signal": range(520)})
        path = service.config.allowed_input_directories[0] / (name + ".csv")
        data.to_csv(path, index=False)
        ids.append(service.data.register(str(path))["id"])

    async def run():
        with pytest.raises(Failure) as exc:
            await service.finetune(
                ForecastFineTuneRequest(
                    dataset_id=ids[0],
                    validation_dataset_id=ids[1],
                    timestamp="time",
                    targets=["signal"],
                    horizon=8,
                )
            )
        assert exc.value.code == "SPLIT_OVERLAP"

    asyncio.run(run())


def test_anomaly_adaptation_rejects_identical_validation_snapshot(service, monkeypatch):
    from kumo_local_mcp.models import AnomalyFineTuneRequest

    async def available(*args):
        pass

    monkeypatch.setattr(service, "require", available)
    first = register(service, "normal.csv", "signal\n" + "\n".join(str(v) for v in range(128)))
    second = register(service, "normal2.csv", "signal\n" + "\n".join(str(v) for v in range(128)))

    async def run():
        with pytest.raises(Failure) as exc:
            await service.finetune(
                AnomalyFineTuneRequest(
                    dataset_id=first["id"], validation_dataset_id=second["id"], columns=["signal"]
                )
            )
        assert exc.value.code == "SPLIT_OVERLAP"

    asyncio.run(run())
