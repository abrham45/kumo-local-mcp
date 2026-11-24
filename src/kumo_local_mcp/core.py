from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Failure(Exception):
    def __init__(self, code: str, message: str, action: str = "Check the tool arguments."):
        super().__init__(message)
        self.code, self.message, self.action = code, message, action

    def response(self):
        return {
            "status": "failed",
            "error": {"code": self.code, "message": self.message, "next_action": self.action},
            "warnings": [],
        }


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Disclosure(StrictModel):
    column_names: bool = False
    summaries: bool = False
    sample_values: bool = False


class Config(StrictModel):
    allowed_input_directories: list[Path] = Field(min_length=1)
    output_directory: Path
    enabled_backends: list[Literal["tabular", "relational", "forecast", "anomaly", "engine"]] = []
    backend_python: dict[str, Path] = {}
    engine_docker_image: str | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    max_queued_jobs: int = Field(default=32, ge=1, le=1000)
    idle_unload_seconds: int = Field(default=300, ge=1)
    job_timeout_seconds: int = Field(default=3600, ge=1)
    max_input_bytes: int = Field(default=100_000_000, ge=1)
    max_rows: int = Field(default=200_000, ge=1)
    max_preview_rows: int = Field(default=10, ge=1, le=100)
    disclosure: Disclosure = Field(default_factory=Disclosure)
    automatic_model_downloads: bool = False
    model_cache_directory: Path | None = None

    @model_validator(mode="after")
    def paths(self):
        self.allowed_input_directories = [
            p.expanduser().resolve() for p in self.allowed_input_directories
        ]
        self.output_directory = self.output_directory.expanduser().resolve()
        if self.model_cache_directory is not None:
            self.model_cache_directory = self.model_cache_directory.expanduser().resolve()
        # Keep the venv interpreter symlink: resolving it would bypass that environment.
        self.backend_python = {k: p.expanduser().absolute() for k, p in self.backend_python.items()}
        return self

    @classmethod
    def load(cls, path: Path):
        try:
            return cls.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            raise Failure(
                "CONFIG_INVALID",
                "Configuration is missing or invalid.",
                "Run kumo-local-mcp setup, then restart serve.",
            ) from exc

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, self.model_dump(mode="json"))


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def atomic_json(path: Path, data):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as f:
            os.chmod(temp, 0o600)
            json.dump(data, f, allow_nan=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def within(path: Path, roots: list[Path], *, exists=True) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=exists)
    except (OSError, RuntimeError) as exc:
        raise Failure("PATH_INVALID", "The local path cannot be resolved.") from exc
    if not any(resolved.is_relative_to(root) for root in roots):
        raise Failure(
            "PATH_DENIED",
            "The path is outside the configured directories.",
            "Choose a permitted path or update configuration locally with setup.",
        )
    return resolved


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Store:
    """Persistent local object registry; apply disclosure policy before returning objects."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root, self._lock = root, threading.RLock()
        self.db = sqlite3.connect(root / "registry.sqlite3", check_same_thread=False)
        os.chmod(root / "registry.sqlite3", 0o600)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT, body TEXT)"
        )
        self.db.commit()

    def put(self, kind: str, body: dict, object_id: str | None = None) -> dict:
        with self._lock:
            body = {**body, "id": object_id or uid(kind), "updated_at": time.time()}
            self.db.execute(
                "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                (body["id"], kind, json.dumps(body, allow_nan=False, default=str)),
            )
            self.db.commit()
            return body

    def get(self, object_id: str, kind: str | None = None) -> dict:
        with self._lock:
            row = self.db.execute(
                "SELECT kind, body FROM objects WHERE id = ?", (object_id,)
            ).fetchone()
        if row is None or (kind and row[0] != kind):
            raise Failure("NOT_FOUND", "The requested local object does not exist.")
        return json.loads(row[1])

    def list(self, kind: str) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT body FROM objects WHERE kind = ?", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def update(self, object_id: str, **changes):
        with self._lock:
            old = self.get(object_id)
            kind = self.db.execute(
                "SELECT kind FROM objects WHERE id = ?", (object_id,)
            ).fetchone()[0]
            return self.put(kind, {**old, **changes}, object_id)

    def close(self):
        self.db.close()
