"""Real-model MCP smoke test. Downloads weights only with --allow-downloads."""

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

from kumo_local_mcp.core import Config, Disclosure


async def main(args):
    work = args.workdir.resolve()
    inputs = work / "data"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 4))
    df = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    df["target"] = (x[:, 0] + x[:, 1] > 0).astype(int)
    df.iloc[:80].to_csv(inputs / "context.csv", index=False)
    df.iloc[80:].to_csv(inputs / "query.csv", index=False)
    history = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=560, freq="h"),
            "signal": np.sin(np.arange(560) / 20) + rng.normal(scale=0.1, size=560),
            "covariate": np.cos(np.arange(560) / 30),
        }
    )
    history.to_csv(inputs / "history.csv", index=False)
    sensors = history[["signal", "covariate"]].copy()
    sensors.loc[300:305, "signal"] += 10
    sensors["label"] = 0
    sensors.loc[300:305, "label"] = 1
    sensors.to_csv(inputs / "sensors.csv", index=False)
    users = pd.DataFrame({"id": range(120), "age": rng.integers(18, 80, 120)})
    users.to_csv(inputs / "users.csv", index=False)
    orders = pd.DataFrame(
        {
            "order_id": range(1200),
            "user_id": rng.integers(0, 120, 1200),
            "amount": rng.uniform(1, 100, 1200),
            "time": pd.date_range("2024-01-01", periods=1200, freq="h"),
        }
    )
    orders.to_csv(inputs / "orders.csv", index=False)
    tasks = pd.DataFrame({"id": range(120), "target": (users["age"] > 45).astype(int)})
    tasks.iloc[:80].to_csv(inputs / "task_context.csv", index=False)
    tasks.iloc[80:].to_csv(inputs / "task_query.csv", index=False)
    config_path = work / "config.json"
    Config(
        allowed_input_directories=[inputs],
        output_directory=work / "output",
        enabled_backends=args.families,
        automatic_model_downloads=args.allow_downloads,
        model_cache_directory=args.weights_cache,
        disclosure=Disclosure(column_names=True, summaries=True),
        job_timeout_seconds=args.timeout,
    ).save(config_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kumo_local_mcp.cli", "--config", str(config_path), "serve"],
        env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    results = {}
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            await client.initialize()

            async def call(name, arguments):
                result = await client.call_tool(name, arguments)
                body = json.loads(result.content[0].text)
                if body.get("status") == "failed":
                    raise RuntimeError(json.dumps(body))
                return body

            async def complete(response):
                deadline = time.monotonic() + args.timeout + 30
                while time.monotonic() < deadline:
                    job = (await call("get_job", {"job_id": response["job_id"]}))["job"]
                    if job["state"] in {"completed", "succeeded"}:
                        return await call("get_result_summary", {"result_id": job["result_id"]})
                    if job["state"] in {"failed", "cancelled", "interrupted"}:
                        raise RuntimeError(json.dumps(job))
                    await asyncio.sleep(0.5)
                raise TimeoutError("Smoke job did not finish before its deadline")

            ids = {}
            for file in inputs.glob("*.csv"):
                ids[file.stem] = (await call("register_dataset", {"path": str(file)}))["dataset_id"]
            capabilities = await call("list_capabilities", {})
            print(json.dumps(capabilities, indent=2))
            tab = {
                "context_dataset_id": ids["context"],
                "query_dataset_id": ids["query"],
                "target": "target",
                "task": "classification",
            }
            for family in args.families:
                try:
                    if family == "tabular":
                        result = await complete(await call("predict_tabular", {"request": tab}))
                        evaluation = await complete(
                            await call(
                                "evaluate_tabular", {"request": tab, "split": {"method": "holdout"}}
                            )
                        )
                        results[family] = {"prediction": result, "evaluation": evaluation}
                        if args.finetune:
                            tuning = {
                                "family": "tabular",
                                "dataset_id": ids["context"],
                                "validation_dataset_id": ids["query"],
                                "target": "target",
                                "task": "classification",
                                "steps": 1,
                                "context_rows": 32,
                                "query_rows": 16,
                            }
                            results[family]["finetune"] = await complete(
                                await call("start_finetuning", {"request": tuning})
                            )
                        if args.explain:
                            results[family]["explanation"] = await complete(
                                await call("explain_result", {"result_id": result["result_id"]})
                            )
                    elif family in {"relational", "engine"}:
                        tables = {
                            "users": {"dataset_id": ids["users"], "primary_key": "id"},
                            "orders": {"dataset_id": ids["orders"], "primary_key": "order_id"},
                        }
                        if family == "engine":
                            tables["orders"]["time_column"] = "time"
                        else:
                            # Native SDM relational smoke deliberately uses non-temporal features.
                            trimmed = orders.drop(columns="time")
                            trimmed.to_csv(inputs / "orders_features.csv", index=False)
                            tables["orders"]["dataset_id"] = (
                                await call(
                                    "register_dataset",
                                    {"path": str(inputs / "orders_features.csv")},
                                )
                            )["dataset_id"]
                        schema = await call(
                            "create_relational_schema",
                            {
                                "tables": tables,
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
                        if family == "relational":
                            req = {
                                "context_dataset_id": ids["task_context"],
                                "query_dataset_id": ids["task_query"],
                                "target": "target",
                                "task": "classification",
                                "schema_id": schema["schema_id"],
                            }
                            results[family] = await complete(
                                await call("predict_relational", {"request": req})
                            )
                        else:
                            graph = await complete(
                                await call(
                                    "engine_create_graph", {"schema_id": schema["schema_id"]}
                                )
                            )
                            snapshot = await complete(
                                await call(
                                    "engine_create_snapshot", {"graph_id": graph["graph_id"]}
                                )
                            )
                            query = {
                                "snapshot_id": snapshot["snapshot_id"],
                                "predictive_query": "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.id",
                            }
                            generated = await complete(
                                await call("engine_generate_training_data", {"request": query})
                            )
                            trained = await complete(
                                await call(
                                    "engine_train_model", {"request": {**query, "max_epochs": 1}}
                                )
                            )
                            predicted = await complete(
                                await call(
                                    "engine_predict",
                                    {
                                        "request": {
                                            **query,
                                            "checkpoint_id": trained["checkpoint_id"],
                                        }
                                    },
                                )
                            )
                            results[family] = {
                                "training_data": generated,
                                "training": trained,
                                "prediction": predicted,
                            }
                    elif family == "forecast":
                        req = {
                            "dataset_id": ids["history"],
                            "timestamp": "time",
                            "targets": ["signal"],
                            "covariates": ["covariate"],
                            "horizon": 8,
                        }
                        results[family] = {
                            "prediction": await complete(
                                await call("forecast_timeseries", {"request": req})
                            ),
                            "backtest": await complete(
                                await call(
                                    "backtest_forecast",
                                    {
                                        "request": req,
                                        "initial_history": 512,
                                        "folds": 2,
                                        "stride": 8,
                                    },
                                )
                            ),
                        }
                    else:
                        req = {
                            "dataset_id": ids["sensors"],
                            "columns": ["signal", "covariate"],
                            "backend_options": {"num_samples": 1, "num_steps": 2},
                        }
                        results[family] = await complete(
                            await call(
                                "evaluate_anomalies", {"request": req, "label_column": "label"}
                            )
                        )
                except Exception as exc:
                    results[family] = {"status": "failed", "message": str(exc)}
    (work / "smoke-report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 1 if any(r.get("status") == "failed" for r in results.values()) else 0
    print(json.dumps(results, indent=2))
    return 1 if any(v.get("status") == "failed" for v in results.values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path(".kumo/smoke"))
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["tabular", "relational", "forecast", "anomaly", "engine"],
        default=["tabular", "relational", "forecast", "anomaly"],
    )
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument(
        "--weights-cache", type=Path, help="Reuse a setup-populated public model cache"
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--explain", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
