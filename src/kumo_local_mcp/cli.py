from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from .core import Config, Disclosure, Failure


def default_config():
    return Path(
        os.environ.get("KUMO_LOCAL_MCP_CONFIG", "~/.config/kumo-local-mcp/config.json")
    ).expanduser()


def python_in(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_install(command):
    # setup runs locally, outside the assistant tool surface, and displays installer progress.
    print("Installing:", " ".join(str(v) for v in command), file=sys.stderr)
    subprocess.run([str(v) for v in command], check=True, stdout=sys.stderr, stderr=sys.stderr)


def setup(args):
    path = args.config.expanduser().resolve()
    existing = Config.load(path) if path.exists() else None
    roots = args.allow_input or (existing.allowed_input_directories if existing else [Path.cwd()])
    output = args.output or (existing.output_directory if existing else Path.cwd() / "kumo-output")
    for root in roots:
        if not root.expanduser().is_dir():
            raise Failure("CONFIG_INVALID", "Every allowed input directory must already exist.")
    values = existing.model_dump() if existing else {}
    values.update(allowed_input_directories=roots, output_directory=output)
    if args.backend is not None:
        values["enabled_backends"] = args.backend
    if args.device is not None:
        values["device"] = args.device
    if args.engine_docker_image:
        values["engine_docker_image"] = args.engine_docker_image
    if args.allow_downloads:
        values["automatic_model_downloads"] = True
    if args.allow_samples or args.allow_columns or args.allow_summaries:
        disclosure = existing.disclosure.model_dump() if existing else Disclosure().model_dump()
        for flag, field in [
            (args.allow_samples, "sample_values"),
            (args.allow_columns, "column_names"),
            (args.allow_summaries, "summaries"),
        ]:
            if flag:
                disclosure[field] = True
        values["disclosure"] = disclosure
    config = Config.model_validate(values)
    config.output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "free_disk_bytes": shutil.disk_usage(config.output_directory).free,
        "config": str(path),
        "installations": [],
        "warnings": [],
    }
    if args.install:
        if not config.enabled_backends:
            raise Failure(
                "CONFIG_INVALID", "Select at least one backend with --backend before installing."
            )
        groups = [
            (
                "sdm",
                [f for f in config.enabled_backends if f != "engine"],
                [
                    "structured-data-models==0.3.0",
                    "pandas>=2.2,<4",
                    "pydantic>=2.10,<3",
                    "scikit-learn>=1.5",
                    "safetensors>=0.5",
                ],
            ),
            (
                "engine",
                ["engine"]
                if "engine" in config.enabled_backends and not config.engine_docker_image
                else [],
                ["kumo-engine==1.0.1", "pandas>=2.2,<4", "pydantic>=2.10,<3", "pyarrow>=15"],
            ),
        ]
        for group, families, requirements in groups:
            if not families:
                continue
            environment = path.parent / "environments" / group
            try:
                if not python_in(environment).exists():
                    if args.python:
                        run_install([args.python, "-m", "venv", environment])
                    else:
                        venv.EnvBuilder(with_pip=True).create(environment)
                run_install(
                    [python_in(environment), "-m", "pip", "install", "--upgrade", *requirements]
                )
                for family in families:
                    config.backend_python[family] = python_in(environment).absolute()
                report["installations"].append({"environment": group, "status": "installed"})
            except (OSError, subprocess.CalledProcessError):
                report["installations"].append({"environment": group, "status": "failed"})
                report["warnings"].append(
                    "Installation failed; inspect the installer output and rerun setup --install."
                )
    config.save(path)
    if args.download_models:
        from .jobs import worker_environment

        cache = config.model_cache_directory or config.output_directory / ".kumo" / "weights"
        for family in config.enabled_backends:
            if family == "engine":
                continue
            interpreter = config.backend_python.get(family, Path(sys.executable))
            command = [
                str(interpreter),
                "-m",
                "kumo_local_mcp.download",
                "--cache",
                str(cache),
                "--family",
                family,
                "--size",
                args.model_size,
                "--workers",
                str(args.download_workers),
            ]
            environment = worker_environment(config)
            environment["HF_HUB_OFFLINE"] = "0"
            completed = subprocess.run(
                command, env=environment, stdout=sys.stderr, stderr=sys.stderr
            )
            if completed.returncode:
                report["installations"].append(
                    {"environment": family + "_weights", "status": "failed"}
                )

    # Probe actual accelerator availability in the selected environment rather than guessing from OS.
    async def check():
        from .jobs import Worker

        capabilities = []
        for family in config.enabled_backends:
            worker = Worker(config, family)
            try:
                response = await worker.call({"operation": "probe", "family": family})
                capabilities.append({"family": family, **response})
            finally:
                await worker.stop()
        return capabilities

    report["backends"] = asyncio.run(check())
    report["status"] = (
        "completed"
        if (
            all(i["status"] != "failed" for i in report["installations"])
            and all(b.get("available", False) for b in report["backends"])
        )
        else "failed"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kumo-local-mcp")
    parser.add_argument(
        "--config", type=Path, default=default_config(), help="Local JSON configuration path"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "setup", help="Check hardware, configure access, and optionally install isolated backends"
    )
    init.add_argument("--allow-input", type=Path, action="append")
    init.add_argument("--output", type=Path)
    init.add_argument(
        "--backend",
        choices=["tabular", "relational", "forecast", "anomaly", "engine"],
        action="append",
    )
    init.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    init.add_argument(
        "--engine-docker-image", help="Use an already-built local Linux/amd64 Engine image"
    )
    init.add_argument(
        "--python", type=Path, help="Python interpreter to use for new worker environments"
    )
    init.add_argument(
        "--install",
        action="store_true",
        help="Download backend packages into isolated environments",
    )
    init.add_argument(
        "--download-models",
        action="store_true",
        help="Prefetch and verify selected model weights; keep serve offline",
    )
    init.add_argument("--download-workers", type=int, choices=range(1, 33), default=8)
    init.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
    init.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Permit first-use Hugging Face weight downloads",
    )
    init.add_argument("--allow-columns", action="store_true")
    init.add_argument("--allow-summaries", action="store_true")
    init.add_argument("--allow-samples", action="store_true")
    sub.add_parser("serve", help="Run the local stdio MCP server")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    try:
        if args.command == "setup":
            return setup(args)
        config = Config.load(args.config)
        # One server owns a registry at a time, so restart recovery cannot interrupt another server.
        from filelock import FileLock, Timeout

        lock = FileLock(config.output_directory / ".kumo" / "server.lock")
        Path(lock.lock_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with lock.acquire(timeout=0):
                from .server import create_server

                server, _ = create_server(config)
                server.run(transport="stdio")
        except Timeout:
            raise Failure(
                "SERVER_BUSY",
                "Another server already owns this output directory.",
                "Use the existing server or choose another output directory.",
            ) from None
        return 0
    except Failure as exc:
        print(json.dumps(exc.response()), file=sys.stderr)
        return 1
    except (OSError, ValueError):
        logging.exception("Startup or setup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
