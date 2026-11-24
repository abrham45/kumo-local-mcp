"""Exercise real Kumo workflows over MCP with a pre-populated offline weight cache."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kumo_local_mcp.core import Config, Disclosure, atomic_json


def make_data(root):
    root.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(417)
    x = random.normal(size=(120, 4))
    table = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    table["target"] = (x[:, 0] + x[:, 1] > 0).astype(int)
    values = {"context": table.iloc[:80], "validation": table.iloc[80:]}
    regression = table.copy()
    regression["target"] = 2 * x[:, 0] - x[:, 1] + 0.1 * random.normal(size=120)
    values.update(reg_context=regression.iloc[:80], reg_validation=regression.iloc[80:])
    users = pd.DataFrame({"id": range(120), "age": random.integers(18, 80, 120)})
    orders = pd.DataFrame(
        {
            "order_id": range(480),
            "user_id": np.repeat(np.arange(120), 4),
            "amount": random.uniform(1, 100, 480),
        }
    )
    tasks = pd.DataFrame({"id": range(120), "target": (users.age > 45).astype(int)})
    reg_tasks = tasks.assign(target=users.age.astype(float) * 2)
    values.update(
        users=users,
        orders=orders,
        task_context=tasks.iloc[:80],
        task_validation=tasks.iloc[80:],
        reg_task_context=reg_tasks.iloc[:80],
        reg_task_validation=reg_tasks.iloc[80:],
    )
    n = np.arange(1120)
    history = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=1120, freq="h"),
            "signal": np.sin(n / 20) + random.normal(0, 0.05, 1120),
            "covariate": np.cos(n / 30),
        }
    )
    history["flag"] = (n % 2).astype(int)
    missing_history = history.iloc[:560].copy()
    missing_history.loc[20:22, "signal"] = np.nan
    normal = history[["signal", "covariate", "flag"]].iloc[:128].copy()
    sensors = normal.copy()
    sensors["label"] = 0
    sensors.loc[60:65, "signal"] += 10
    sensors.loc[60:65, "label"] = 1
    values.update(
        history=history.iloc[:560],
        missing_history=missing_history,
        history_validation=history.iloc[560:],
        sensors=sensors,
        normal=normal,
        normal_validation=history[["signal", "covariate", "flag"]].iloc[200:328],
    )
    engine_orders = pd.DataFrame(
        {
            "order_id": range(7200),
            "user_id": np.tile(np.arange(60), 120),
            "amount": random.uniform(1, 100, 7200),
            "time": np.repeat(pd.date_range("2024-01-01", periods=120, freq="D"), 60),
        }
    )
    values["engine_orders"] = engine_orders
    for name, frame in values.items():
        frame.to_csv(root / (name + ".csv"), index=False)


async def verify(args):
    work = args.workdir.resolve()
    make_data(work / "data")
    config = Config(
        allowed_input_directories=[work / "data"],
        output_directory=work / "output",
        model_cache_directory=args.weights_cache.resolve(),
        enabled_backends=args.families,
        disclosure=Disclosure(column_names=True, summaries=True, sample_values=True),
        job_timeout_seconds=args.timeout,
        device="cpu",
    )
    if args.engine_python:
        config.backend_python["engine"] = args.engine_python.absolute()
        config.device = "auto"
    if args.engine_docker_image:
        config.engine_docker_image = args.engine_docker_image
        config.device = "auto"
    config.save(work / "config.json")
    report = {"status": "running", "checks": [], "versions": {}}
    started = time.monotonic()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kumo_local_mcp.cli", "--config", str(work / "config.json"), "serve"],
        env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            await client.initialize()

            async def call(tool, body):
                response = await client.call_tool(tool, body)
                result = json.loads(response.content[0].text)
                if response.isError or result.get("status") == "failed":
                    raise RuntimeError(json.dumps({"tool": tool, "response": result}))
                return result

            async def finish(queued):
                deadline = time.monotonic() + args.timeout + 30
                while time.monotonic() < deadline:
                    job = (await call("get_job_status", {"job_id": queued["job_id"]}))["job"]
                    if job["state"] in {"completed", "succeeded"}:
                        return await call("get_result_summary", {"result_id": job["result_id"]})
                    if job["state"] in {"failed", "cancelled", "interrupted"}:
                        raise RuntimeError(json.dumps(job))
                    await asyncio.sleep(0.25)
                raise TimeoutError("Integration job deadline exceeded")

            async def operation(tool, body):
                return await finish(await call(tool, body))

            async def check(label, coroutine):
                before = time.monotonic()
                try:
                    result = await coroutine
                    entry = {"check": label, "status": "passed", "result": result}
                except Exception as error:
                    entry = {"check": label, "status": "failed", "error": str(error)}
                    result = None
                entry["seconds"] = round(time.monotonic() - before, 3)
                report["checks"].append(entry)
                atomic_json(work / "verification.json", report)
                print(label, entry["status"], entry["seconds"], flush=True)
                return result

            ids = {}
            for path in (work / "data").glob("*.csv"):
                ids[path.stem] = (await call("register_dataset", {"path": str(path)}))["dataset_id"]
            report["versions"] = await call("list_capabilities", {})

            async def check_outputs(result):
                assert result and Path(result["output_file"]).is_file()
                frame = pd.read_csv(result["output_file"])
                assert len(frame) > 0
                assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
                if result["family"] in {"tabular", "relational"}:
                    values = frame.to_numpy(dtype=float)
                    if "q500" in frame:
                        assert frame.shape[1] == 999
                        assert (np.diff(values, axis=1) >= -1e-6).all()
                    else:
                        assert (values >= 0).all() and (values <= 1).all()
                        assert np.allclose(values.sum(axis=1), 1, atol=1e-5)
                if result["family"] == "anomaly":
                    assert (frame["mae"] >= 0).all()
                preview = await call(
                    "preview_result", {"result_id": result["result_id"], "limit": 2}
                )
                assert 0 < len(preview["rows"]) <= 2
                resource = await client.read_resource(result["resource_uri"])
                assert json.loads(resource.contents[0].text) == result
                exports = []
                for format in ("csv", "parquet", "json"):
                    export = await call(
                        "export_result",
                        {
                            "result_id": result["result_id"],
                            "filename": "exports/" + result["result_id"] + "." + format,
                            "format": format,
                        },
                    )
                    assert Path(export["path"]).stat().st_size > 0
                    exported = (
                        pd.read_csv(export["path"])
                        if format == "csv"
                        else pd.read_parquet(export["path"])
                        if format == "parquet"
                        else pd.read_json(export["path"])
                    )
                    assert exported.shape == frame.shape
                    exports.append(export["path"])
                return {"preview_rows": len(preview["rows"]), "exports": exports}

            async def session_check(family, request, prediction_tool):
                options = request.get("backend_options", {})
                body = {"family": family, "backend_options": options}
                if family == "tabular":
                    body.update(
                        context_dataset_id=request["context_dataset_id"],
                        target="target",
                        task=request["task"],
                    )
                loaded = await call("create_session", body)
                await finish(loaded)
                first = await operation(
                    prediction_tool, {"request": {**request, "session_id": loaded["session_id"]}}
                )
                second = await operation(
                    prediction_tool, {"request": {**request, "session_id": loaded["session_id"]}}
                )
                sessions = (await call("list_sessions", {}))["sessions"]
                assert any(
                    s["session_id"] == loaded["session_id"] and s["worker_resident"]
                    for s in sessions
                )
                await call("close_session", {"session_id": loaded["session_id"]})
                return {
                    "session_id": loaded["session_id"],
                    "results": [first["result_id"], second["result_id"]],
                }

            for family in args.families:
                if family in {"tabular", "relational"}:
                    schema = None
                    if family == "relational":
                        schema = await call(
                            "create_relational_schema",
                            {
                                "tables": {
                                    "users": {"dataset_id": ids["users"], "primary_key": "id"},
                                    "orders": {
                                        "dataset_id": ids["orders"],
                                        "primary_key": "order_id",
                                    },
                                },
                                "relationships": [
                                    {
                                        "from_table": "orders",
                                        "from_column": "user_id",
                                        "to_table": "users",
                                        "to_column": "id",
                                    }
                                ],
                                "task_link": {
                                    "task_column": "id",
                                    "table": "users",
                                    "table_column": "id",
                                },
                            },
                        )
                    for task in args.tasks:
                        stem = "reg_" if task == "regression" else ""
                        stem += "task_" if family == "relational" else ""
                        req = {
                            "context_dataset_id": ids[stem + "context"],
                            "query_dataset_id": ids[stem + "validation"],
                            "target": "target",
                            "task": task,
                        }
                        if schema:
                            req["schema_id"] = schema["schema_id"]
                        result = await check(
                            f"{family}.{task}.prediction",
                            operation("predict_" + family, {"request": req}),
                        )
                        await check(
                            f"{family}.{task}.evaluation",
                            operation(
                                "evaluate_" + family,
                                {"request": req, "split": {"method": "holdout"}},
                            ),
                        )
                        if result:
                            await check(f"{family}.{task}.outputs", check_outputs(result))
                            await check(
                                f"{family}.{task}.explanation",
                                operation("explain_result", {"result_id": result["result_id"]}),
                            )
                        tune = {
                            "family": family,
                            "dataset_id": req["context_dataset_id"],
                            "validation_dataset_id": req["query_dataset_id"],
                            "target": "target",
                            "task": task,
                            "steps": 1,
                            "context_rows": 32,
                            "query_rows": 16,
                        }
                        if schema:
                            tune["schema_id"] = schema["schema_id"]
                        tuned = await check(
                            f"{family}.{task}.finetune",
                            operation("start_finetuning", {"request": tune}),
                        )
                        if tuned:
                            await check(
                                f"{family}.{task}.checkpoint",
                                operation(
                                    "predict_" + family,
                                    {"request": {**req, "checkpoint_id": tuned["checkpoint_id"]}},
                                ),
                            )
                        if family == "tabular":
                            await check(
                                f"{family}.{task}.session",
                                session_check(family, req, "predict_tabular"),
                            )
                elif family == "forecast":
                    req = {
                        "dataset_id": ids["history"],
                        "timestamp": "time",
                        "targets": ["signal"],
                        "covariates": ["covariate"],
                        "horizon": 8,
                    }
                    result = await check(
                        "forecast.prediction", operation("forecast_timeseries", {"request": req})
                    )
                    await check(
                        "forecast.backtest",
                        operation(
                            "backtest_forecast",
                            {"request": req, "initial_history": 512, "folds": 2, "stride": 8},
                        ),
                    )
                    if result:
                        await check("forecast.outputs", check_outputs(result))
                    multi = await check(
                        "forecast.multitarget_integer_covariate",
                        operation(
                            "forecast_timeseries",
                            {
                                "request": {
                                    **req,
                                    "targets": ["signal", "covariate"],
                                    "covariates": ["flag"],
                                }
                            },
                        ),
                    )
                    if multi:
                        await check("forecast.multitarget_outputs", check_outputs(multi))
                    await check(
                        "forecast.missing_interpolation",
                        operation(
                            "forecast_timeseries",
                            {
                                "request": {
                                    **req,
                                    "dataset_id": ids["missing_history"],
                                    "missing_values": "interpolate",
                                }
                            },
                        ),
                    )
                    await check(
                        "forecast.session", session_check(family, req, "forecast_timeseries")
                    )
                    tune = {
                        **req,
                        "family": "forecast",
                        "validation_dataset_id": ids["history_validation"],
                        "steps": 1,
                    }
                    tuned = await check(
                        "forecast.finetune", operation("start_finetuning", {"request": tune})
                    )
                    if tuned:
                        await check(
                            "forecast.checkpoint",
                            operation(
                                "forecast_timeseries",
                                {"request": {**req, "checkpoint_id": tuned["checkpoint_id"]}},
                            ),
                        )
                elif family == "anomaly":
                    req = {
                        "dataset_id": ids["sensors"],
                        "columns": ["signal", "covariate", "flag"],
                        "threshold": 0,
                        "backend_options": {"num_samples": 1, "num_steps": 2},
                    }
                    result = await check(
                        "anomaly.prediction", operation("detect_anomalies", {"request": req})
                    )
                    await check(
                        "anomaly.labeled_evaluation",
                        operation("evaluate_anomalies", {"request": req, "label_column": "label"}),
                    )
                    await check(
                        "anomaly.unlabeled_diagnostics",
                        operation("evaluate_anomalies", {"request": req}),
                    )
                    if result:
                        await check("anomaly.outputs", check_outputs(result))
                    await check("anomaly.session", session_check(family, req, "detect_anomalies"))
                    tune = {
                        **req,
                        "dataset_id": ids["normal"],
                        "validation_dataset_id": ids["normal_validation"],
                        "family": "anomaly",
                        "steps": 1,
                        "window_rows": 128,
                    }
                    tuned = await check(
                        "anomaly.finetune", operation("start_finetuning", {"request": tune})
                    )
                    if tuned:
                        await check(
                            "anomaly.checkpoint",
                            operation(
                                "detect_anomalies",
                                {"request": {**req, "checkpoint_id": tuned["checkpoint_id"]}},
                            ),
                        )
                else:
                    schema = await call(
                        "create_relational_schema",
                        {
                            "tables": {
                                "users": {"dataset_id": ids["users"], "primary_key": "id"},
                                "orders": {
                                    "dataset_id": ids["engine_orders"],
                                    "primary_key": "order_id",
                                    "time_column": "time",
                                },
                            },
                            "relationships": [
                                {
                                    "from_table": "orders",
                                    "from_column": "user_id",
                                    "to_table": "users",
                                    "to_column": "id",
                                }
                            ],
                        },
                    )
                    graph = await check(
                        "engine.graph",
                        operation("engine_create_graph", {"schema_id": schema["schema_id"]}),
                    )
                    if not graph:
                        continue
                    snap = await check(
                        "engine.snapshot",
                        operation("engine_create_snapshot", {"graph_id": graph["graph_id"]}),
                    )
                    if not snap:
                        continue
                    query = {
                        "snapshot_id": snap["snapshot_id"],
                        "predictive_query": "PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.id",
                    }
                    await check(
                        "engine.training_data",
                        operation("engine_generate_training_data", {"request": query}),
                    )
                    trained = await check(
                        "engine.train",
                        operation(
                            "engine_train_model",
                            {
                                "request": {
                                    **query,
                                    "max_epochs": 1,
                                    "max_steps_per_epoch": 2,
                                    "max_eval_steps": 2,
                                }
                            },
                        ),
                    )
                    if trained:
                        predicted = await check(
                            "engine.prediction",
                            operation(
                                "engine_predict",
                                {"request": {**query, "checkpoint_id": trained["checkpoint_id"]}},
                            ),
                        )
                        if predicted:
                            await check("engine.outputs", check_outputs(predicted))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["status"] = (
        "failed" if any(c["status"] == "failed" for c in report["checks"]) else "passed"
    )
    atomic_json(work / "verification.json", report)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["tabular", "relational", "forecast", "anomaly", "engine"],
        required=True,
    )
    parser.add_argument("--weights-cache", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["classification", "regression"],
        default=["classification", "regression"],
    )
    parser.add_argument("--engine-python", type=Path)
    parser.add_argument("--engine-docker-image")
    raise SystemExit(asyncio.run(verify(parser.parse_args())))
