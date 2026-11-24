import asyncio
import sys

from kumo_local_mcp.core import Config, Store
from kumo_local_mcp.jobs import JobManager, Worker


class SleepingWorker(Worker):
    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            "import sys,time; sys.stdin.readline(); time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )


def test_running_cancellation_kills_worker_and_records_terminal_state(tmp_path):
    async def run():
        config = Config(allowed_input_directories=[tmp_path], output_directory=tmp_path / "out")
        store = Store(config.output_directory / ".kumo")
        jobs = JobManager(config, store)
        worker = SleepingWorker(config, "tabular")
        jobs.workers["tabular"] = worker
        response = jobs.submit("tabular", "predict", {})
        for _ in range(100):
            if worker.process:
                break
            await asyncio.sleep(0.01)
        process = worker.process
        assert process is not None
        directory = next((store.root / "results").iterdir())
        (directory / "unfinished.tmp").write_text("partial")
        cancelled = await jobs.cancel(response["job_id"])
        assert cancelled["partial_output_directory"] == str(directory)
        assert cancelled["warnings"]
        assert process.returncode is not None
        assert store.get(response["job_id"])["state"] == "cancelled"
        await jobs.close()
        store.close()

    asyncio.run(run())


def test_worker_crash_is_actionable_and_does_not_crash_manager(tmp_path):
    class CrashingWorker(SleepingWorker):
        async def start(self):
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "raise SystemExit(7)",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

    async def run():
        config = Config(allowed_input_directories=[tmp_path], output_directory=tmp_path / "out")
        store = Store(config.output_directory / ".kumo")
        jobs = JobManager(config, store)
        jobs.workers["tabular"] = CrashingWorker(config, "tabular")
        response = jobs.submit("tabular", "predict", {})
        await jobs.tasks[response["job_id"]]
        result = store.get(response["job_id"])
        assert result["state"] == "failed"
        assert result["error"]["code"] in {"WORKER_CRASHED", "WORKER_PROTOCOL"}
        await jobs.close()
        store.close()

    asyncio.run(run())


def test_timeout_terminates_worker(tmp_path):
    async def run():
        config = Config(
            allowed_input_directories=[tmp_path],
            output_directory=tmp_path / "out",
            job_timeout_seconds=1,
        )
        store = Store(config.output_directory / ".kumo")
        jobs = JobManager(config, store)
        worker = SleepingWorker(config, "tabular")
        jobs.workers["tabular"] = worker
        response = jobs.submit("tabular", "predict", {})
        await jobs.tasks[response["job_id"]]
        assert store.get(response["job_id"])["error"]["code"] == "JOB_TIMEOUT"
        assert worker.process is None
        await jobs.close()
        store.close()

    asyncio.run(run())


def test_cancellation_before_task_first_runs_is_still_a_normal_response(tmp_path):
    async def run():
        config = Config(allowed_input_directories=[tmp_path], output_directory=tmp_path / "out")
        store = Store(config.output_directory / ".kumo")
        jobs = JobManager(config, store)
        response = jobs.submit("tabular", "predict", {})
        cancelled = await jobs.cancel(response["job_id"])
        assert cancelled["status"] == "completed" and cancelled["state"] == "cancelled"
        assert not jobs.workers
        await jobs.close()
        store.close()

    asyncio.run(run())
