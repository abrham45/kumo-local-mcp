from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from .core import Config, Failure, Store, atomic_json, uid

log = logging.getLogger(__name__)


def worker_environment(config: Config):
    # Do not forward database credentials, API keys, or unrelated app secrets.
    keep = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "HOME",
        "USERPROFILE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["HF_HOME"] = str(
        config.model_cache_directory or config.output_directory / ".kumo" / "weights"
    )
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["HF_HUB_OFFLINE"] = "0" if config.automatic_model_downloads else "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


class Worker:
    def __init__(self, config: Config, family: str):
        self.config, self.family = config, family
        self.lock = asyncio.Lock()
        self.process = None
        self.container_name = None
        self.last_used = time.monotonic()

    async def start(self):
        if self.process is not None and self.process.returncode is None:
            return
        python = self.config.backend_python.get(self.family, Path(sys.executable))
        environment = worker_environment(self.config)
        command = [str(python), "-m", "kumo_local_mcp.worker"]
        if self.family == "engine" and self.config.engine_docker_image:
            docker = shutil.which("docker")
            if not docker:
                raise Failure(
                    "ENGINE_RUNTIME_MISSING",
                    "Docker is required by the configured Engine runtime.",
                    "Start Docker and build the documented local Engine image.",
                )
            self.container_name = "kumo-mcp-" + uid("worker").removeprefix("worker_")
            package = Path(__file__).resolve().parent
            output = self.config.output_directory
            command = [
                docker,
                "run",
                "--rm",
                "--init",
                "--interactive",
                "--pull=never",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp",
                # Engine's shared-memory policy selects inline loading at this limit.
                # This avoids pidfd-dependent worker pools unsupported by amd64 emulation.
                "--shm-size=64m",
                "--name",
                self.container_name,
                "--label",
                "kumo-local-mcp.worker=true",
                "--volume",
                f"{package}:/opt/kumo-mcp/kumo_local_mcp:ro",
                "--volume",
                f"{output}:{output}",
                "--workdir",
                str(output),
                "--env",
                "PYTHONPATH=/opt/kumo-mcp",
                "--env",
                "HOME=/tmp",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "--env",
                "DO_NOT_TRACK=1",
                "--env",
                "OMP_NUM_THREADS=2",
            ]
            if os.name != "nt":
                command += ["--user", f"{os.getuid()}:{os.getgid()}"]
            command += [self.config.engine_docker_image, "python", "-m", "kumo_local_mcp.worker"]
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            env=environment,
            cwd=self.config.output_directory,
            start_new_session=(os.name != "nt"),
            limit=2**20,
        )

    async def stop(self):
        p, self.process = self.process, None
        container, self.container_name = self.container_name, None
        if container:
            # Killing a Docker CLI alone can leave expensive container work alive.
            try:
                cleanup = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "--force",
                    container,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(cleanup.wait(), 15)
                except TimeoutError:
                    cleanup.kill()
                    await cleanup.wait()
            except OSError:
                log.warning("Docker cleanup could not start for this worker")
            if p is not None and p.returncode is None:
                # Removing the container normally exits its attached Docker CLI asynchronously.
                # Reap it before attempting any host process-group signal.
                try:
                    await asyncio.wait_for(p.wait(), 3)
                    return
                except TimeoutError:
                    pass
        if p is None or p.returncode is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(p.pid, signal.SIGTERM)
            else:
                p.terminate()
            try:
                await asyncio.wait_for(p.wait(), 3)
            except TimeoutError:
                if os.name != "nt":
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    p.kill()
                await p.wait()
        except ProcessLookupError:
            pass

    async def call(self, payload: dict):
        async with self.lock:
            try:
                await self.start()
                self.process.stdin.write((json.dumps(payload) + "\n").encode())
                await self.process.stdin.drain()
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), self.config.job_timeout_seconds
                )
                if not line:
                    raise Failure(
                        "WORKER_CRASHED",
                        "The backend worker exited unexpectedly.",
                        "Check local stderr logs, device memory, and backend installation.",
                    )
                return json.loads(line)
            except (asyncio.CancelledError, TimeoutError):
                await self.stop()
                raise
            except (OSError, ValueError, ConnectionError) as exc:
                await self.stop()
                raise Failure(
                    "WORKER_PROTOCOL",
                    "The local worker could not complete the request.",
                    "Run setup to check or repair the backend environment.",
                ) from exc
            finally:
                self.last_used = time.monotonic()


class JobManager:
    def __init__(self, config: Config, store: Store):
        self.config, self.store = config, store
        self.semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
        self.tasks, self.workers = {}, {}
        self.reaper = None
        for job in store.list("job"):
            if job["state"] in {"queued", "running"}:
                store.update(
                    job["id"],
                    state="interrupted",
                    progress=0,
                    error=Failure(
                        "SERVER_RESTARTED",
                        "The server stopped before this job finished.",
                        "Submit the operation again.",
                    ).response()["error"],
                )
        # Sessions are not durable loaded objects. Old IDs cannot silently refer to new state.
        for session in store.list("session"):
            store.update(session["id"], state="expired")

    def worker(self, key, family):
        if key not in self.workers:
            self.workers[key] = Worker(self.config, family)
        return self.workers[key]

    async def reap(self):
        while True:
            await asyncio.sleep(min(30, self.config.idle_unload_seconds))
            for worker in list(self.workers.values()):
                if (
                    not worker.lock.locked()
                    and time.monotonic() - worker.last_used > self.config.idle_unload_seconds
                ):
                    # Acquire before stopping so a newly arriving job cannot lose its worker.
                    async with worker.lock:
                        await worker.stop()

    def submit(self, family: str, operation: str, payload: dict, session_id=None):
        if sum(not t.done() for t in self.tasks.values()) >= self.config.max_queued_jobs:
            raise Failure(
                "QUEUE_FULL", "The configured job queue is full.", "Wait for a job to finish."
            )
        if self.reaper is None:
            self.reaper = asyncio.create_task(self.reap())
        job_id, result_id = uid("job"), uid("result")
        job = self.store.put(
            "job",
            {
                "family": family,
                "operation": operation,
                "state": "queued",
                "progress": 0,
                "result_id": None,
                "session_id": session_id,
                "created_at": time.time(),
            },
            job_id,
        )
        self.tasks[job_id] = asyncio.create_task(self.run(job, result_id, payload, session_id))
        return {"status": "queued", "job_id": job_id, "warnings": []}

    async def run(self, job, result_id, payload, session_id):
        directory = self.store.root / "results" / result_id
        try:
            async with self.semaphore:
                self.store.update(job["id"], state="running", progress=0.1)
                directory.mkdir(parents=True, mode=0o700)
                request = {
                    **payload,
                    "operation": job["operation"],
                    "family": job["family"],
                    "output": str(directory),
                    "config": self.config.model_dump(mode="json"),
                    "session_id": session_id,
                }
                key = session_id or job["family"]
                # Idle caches must not accumulate one resident model per historical session.
                for old_key, old_worker in list(self.workers.items()):
                    if old_key != key and not old_worker.lock.locked():
                        async with old_worker.lock:
                            await old_worker.stop()
                response = await self.worker(key, job["family"]).call(request)
                if response.get("status") == "failed":
                    self.store.update(
                        job["id"], state="failed", error=response["error"], progress=1
                    )
                    return
                result = self.store.put(
                    "result",
                    {
                        "job_id": job["id"],
                        "family": job["family"],
                        "operation": job["operation"],
                        "directory": str(directory),
                        "request": payload,
                        **response,
                    },
                    result_id,
                )
                if response.get("checkpoint"):
                    checkpoint = self.store.put(
                        "checkpoint",
                        {
                            "family": job["family"],
                            "path": response["checkpoint"],
                            "settings": response.get("model_settings", {}),
                            "result_id": result_id,
                        },
                    )
                    self.store.update(result_id, checkpoint_id=checkpoint["id"])
                if job["family"] == "engine" and job["operation"] in {"graph", "snapshot"}:
                    kind = "graph" if job["operation"] == "graph" else "snapshot"
                    item = self.store.put(
                        kind, {"schema_id": payload["schema_id"], "result_id": result_id}
                    )
                    self.store.update(result_id, **{kind + "_id": item["id"]})
                result = self.store.get(result_id, "result")
                atomic_json(directory / "metadata.json", result)
                self.store.update(job["id"], state="completed", result_id=result_id, progress=1)
        except asyncio.CancelledError:
            self.store.update(job["id"], state="cancelled", progress=1)
        except TimeoutError:
            self.store.update(
                job["id"],
                state="failed",
                progress=1,
                error=Failure(
                    "JOB_TIMEOUT",
                    "The backend exceeded the configured execution timeout.",
                    "Use a smaller workload or increase the timeout locally.",
                ).response()["error"],
            )
        except Failure as exc:
            self.store.update(job["id"], state="failed", error=exc.response()["error"], progress=1)
        except Exception:
            log.exception("Job %s failed", job["id"])
            self.store.update(
                job["id"],
                state="failed",
                progress=1,
                error=Failure(
                    "JOB_FAILED",
                    "The backend operation failed.",
                    "Inspect local stderr logs and validate the data and backend.",
                ).response()["error"],
            )
        finally:
            # Leave incomplete artifacts available for local inspection, never as successful results.
            if self.store.get(job["id"])["state"] in {"failed", "cancelled"}:
                if directory.is_dir() and any(directory.iterdir()):
                    self.store.update(job["id"], partial_output_directory=str(directory))

    async def cancel(self, job_id):
        self.store.get(job_id, "job")
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # A task cancelled before its first scheduled instruction never runs its handler.
            self.store.update(job_id, state="cancelled", progress=1)
        item = self.store.get(job_id)
        partial = item.get("partial_output_directory")
        return {
            "status": "completed",
            "job_id": job_id,
            "state": item["state"],
            **({"partial_output_directory": partial} if partial else {}),
            "warnings": ["Incomplete local artifacts are not valid results."] if partial else [],
        }

    async def close(self):
        if self.reaper:
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        await asyncio.gather(*(w.stop() for w in self.workers.values()))
