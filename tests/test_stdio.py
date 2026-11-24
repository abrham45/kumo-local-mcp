import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kumo_local_mcp.core import Config, Store


def test_real_stdio_handshake_tools_data_and_filtered_resources(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    csv = inputs / "source.csv"
    csv.write_text("sensitive_name,target\n1,0\n2,1\n")
    config = Config(allowed_input_directories=[inputs], output_directory=tmp_path / "out")
    path = tmp_path / "config.json"
    config.save(path)
    store = Store(config.output_directory / ".kumo")
    saved = store.put(
        "result",
        {
            "family": "tabular",
            "operation": "predict",
            "warnings": [],
            "request": {"target": "sensitive_name", "secret": "DO-NOT-RETURN"},
            "metrics": {"accuracy": 0.9},
        },
    )
    job = store.put("job", {"family": "tabular", "state": "completed", "progress": 1})
    store.close()

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "kumo_local_mcp.cli", "--config", str(path), "serve"],
            env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                assert {
                    "predict_tabular",
                    "predict_relational",
                    "forecast_timeseries",
                    "detect_anomalies",
                    "start_finetuning",
                    "engine_train_model",
                    "cancel_job",
                    "get_job_status",
                    "list_sessions",
                }.issubset(names)
                for name in ("get_job", "get_job_status"):
                    status = await session.call_tool(name, {"job_id": job["id"]})
                    assert json.loads(status.content[0].text)["job"]["state"] == "succeeded"
                sessions = await session.call_tool("list_sessions", {})
                assert json.loads(sessions.content[0].text)["sessions"] == []
                result = await session.call_tool("register_dataset", {"path": str(csv)})
                body = json.loads(result.content[0].text)
                assert body["status"] == "completed" and body["dataset_id"]
                assert "sensitive_name" not in json.dumps(body)
                preview = await session.call_tool(
                    "preview_dataset", {"dataset_id": body["dataset_id"]}
                )
                assert "DISCLOSURE_DENIED" in preview.content[0].text
                cap = await session.call_tool("list_capabilities", {})
                assert all(not b["enabled"] for b in json.loads(cap.content[0].text)["backends"])
                templates = await session.list_resource_templates()
                assert len(templates.resourceTemplates) == 2
                unknown = await session.read_resource("kumo://results/result_missing")
                assert "NOT_FOUND" in unknown.contents[0].text
                resource = await session.read_resource("kumo://results/" + saved["id"])
                assert "sensitive_name" not in resource.contents[0].text
                assert "DO-NOT-RETURN" not in resource.contents[0].text
                assert "accuracy" not in resource.contents[0].text

    asyncio.run(run())
