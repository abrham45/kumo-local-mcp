"""Verified, resumable public-weight transfers, invoked only by local setup."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .core import Failure, atomic_json


def transfer_ranges(
    url: str,
    target: Path,
    size: int,
    sha256: str,
    *,
    workers=8,
    chunk_bytes=1024 * 1024,
    timeout=120,
    attempts=4,
    progress=None,
    refresh_url=None,
):
    """Download immutable bytes. Parts survive interruption; final files require a full digest."""
    if size < 1 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise Failure("MODEL_METADATA_INVALID", "Model size or SHA-256 metadata is invalid.")
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as cached:
            if hashlib.file_digest(cached, "sha256").hexdigest() == sha256:
                return target
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parts = target.with_name(target.name + ".parts")
    parts.mkdir(exist_ok=True, mode=0o700)
    count = (size + chunk_bytes - 1) // chunk_bytes
    current_url = [url]
    refresh_lock = threading.Lock()

    def get_part(index):
        start, end = index * chunk_bytes, min(size, (index + 1) * chunk_bytes) - 1
        expected = end - start + 1
        part = parts / str(index)
        if part.is_file() and part.stat().st_size == expected:
            return expected
        for attempt in range(attempts):
            temporary = part.with_suffix(".tmp")
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > expected:
                temporary.unlink()
                offset = 0
            if offset == expected:
                temporary.replace(part)
                return expected
            request_start = start + offset
            request_url = current_url[0]
            try:
                request = urllib.request.Request(
                    request_url,
                    headers={
                        "Range": f"bytes={request_start}-{end}",
                        "Accept-Encoding": "identity",
                        "User-Agent": "kumo-local-mcp/0.1",
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if (
                        response.status != 206
                        or content_range != f"bytes {request_start}-{end}/{size}"
                    ):
                        raise Failure(
                            "MODEL_RANGE_UNSUPPORTED",
                            "The model endpoint did not honor a bounded range request.",
                        )
                    with temporary.open("ab") as output:
                        remaining = expected - offset
                        while remaining:
                            data = response.read1(min(64 * 1024, remaining))
                            if not data:
                                raise OSError("Incomplete model range")
                            output.write(data)
                            remaining -= len(data)
                    temporary.chmod(0o600)
                    temporary.replace(part)
                return expected
            except Failure:
                raise
            except (OSError, TimeoutError) as error:
                code = getattr(error, "code", None)
                if code in {401, 403} and refresh_url is not None:
                    with refresh_lock:
                        if current_url[0] == request_url:
                            current_url[0] = refresh_url()
                print(
                    f"Model range {index}, attempt {attempt + 1}: {type(error).__name__}"
                    + (f" (HTTP {code})" if code else ""),
                    file=sys.stderr,
                    flush=True,
                )
                if attempt + 1 == attempts:
                    raise Failure(
                        "MODEL_DOWNLOAD_FAILED",
                        "A public model-weight transfer failed.",
                        "Rerun setup --download-models to resume verified download parts.",
                    ) from None
                time.sleep(min(attempt + 1, 3))

    completed = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(get_part, index) for index in range(count)]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            if progress:
                progress(completed, size)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    staging = target.with_name(target.name + ".verified.tmp")
    digest = hashlib.sha256()
    try:
        with staging.open("wb") as output:
            for index in range(count):
                with (parts / str(index)).open("rb") as source:
                    while data := source.read(1024 * 1024):
                        digest.update(data)
                        output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != sha256:
            # Bad parts must not poison every subsequent retry.
            for index in range(count):
                (parts / str(index)).unlink(missing_ok=True)
            raise Failure(
                "MODEL_DIGEST_MISMATCH",
                "Downloaded model weights failed SHA-256 verification.",
                "Retry the download; the unverified model was not installed.",
            )
        staging.chmod(0o600)
        staging.replace(target)
        for index in range(count):
            (parts / str(index)).unlink(missing_ok=True)
        parts.rmdir()
    finally:
        staging.unlink(missing_ok=True)
    return target


def fetch_model(repo: str, filename: str, revision: str, cache: Path, workers=8):
    from filelock import FileLock
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo, filename=filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    etag = metadata.etag
    if not etag or not re.fullmatch(r"[a-f0-9]{64}", etag) or not metadata.commit_hash:
        raise Failure(
            "MODEL_METADATA_INVALID",
            "Expected immutable SHA-256 model metadata from the public repository.",
        )
    directory = cache / "hub" / ("models--" + repo.replace("/", "--"))
    target = directory / "blobs" / etag
    target.parent.mkdir(parents=True, exist_ok=True)
    started, last = time.monotonic(), [0.0]

    def report(done, total):
        now = time.monotonic()
        if now - last[0] >= 10 or done == total:
            print(
                f"{repo}/{filename}: {done / total:.0%} ({done / 1e6:.1f}/{total / 1e6:.1f} MB)",
                file=sys.stderr,
                flush=True,
            )
            last[0] = now

    def refresh():
        latest = get_hf_file_metadata(url + f"?download=true&refresh={time.time_ns()}")
        if (
            latest.etag != etag
            or latest.commit_hash != metadata.commit_hash
            or latest.size != metadata.size
        ):
            raise Failure(
                "MODEL_METADATA_CHANGED", "Model metadata changed during a pinned download."
            )
        return latest.location

    with FileLock(str(target) + ".download.lock"):
        transfer_ranges(
            metadata.location,
            target,
            metadata.size,
            etag,
            workers=workers,
            progress=report,
            refresh_url=refresh,
        )
        snapshot = directory / "snapshots" / metadata.commit_hash / filename
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.symlink_to(os.path.relpath(target, snapshot.parent))
        # SDM's pinned tag resolves offline through this normal Hugging Face cache ref.
        ref = directory / "refs" / revision
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(metadata.commit_hash)
    return {
        "repository": repo,
        "filename": filename,
        "revision": metadata.commit_hash,
        "sha256": etag,
        "bytes": metadata.size,
        "seconds": round(time.monotonic() - started, 2),
    }


def model_files(family, size="small"):
    if family in {"tabular", "relational"}:
        prefix = size + "/" if family == "tabular" else ""
        return [
            ("nvidia/Kumo-" + family.title(), prefix + task + ".pt", "v1.0.1")
            for task in ("classifier", "regressor")
        ]
    if family == "anomaly":
        return [
            ("nvidia/Kumo-Anomaly", "final_model.pth", "226c5003b0582adc1cac433123f47324bfae39f2")
        ]
    return [
        (
            "nvidia/Kumo-Forecast",
            "run8_best_model_cr.pt",
            "abff20a58834638b28227ff4ab934f26206e4b09",
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--family",
        choices=["tabular", "relational", "forecast", "anomaly"],
        action="append",
        required=True,
    )
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--workers", type=int, choices=range(1, 33), default=8)
    args = parser.parse_args()
    records = []
    for family in args.family:
        for repo, file, revision in model_files(family, args.size):
            record = fetch_model(repo, file, revision, args.cache, args.workers)
            records.append(record)
            from filelock import FileLock

            manifest = args.cache / "download-manifest.json"
            with FileLock(str(manifest) + ".lock"):
                saved = (
                    json.loads(manifest.read_text()).get("files", []) if manifest.exists() else []
                )
                saved = [r for r in saved if (r["repository"], r["filename"]) != (repo, file)]
                atomic_json(manifest, {"files": [*saved, record]})
    print(json.dumps({"status": "completed", "files": records}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Failure as exc:
        print(json.dumps(exc.response()), file=sys.stderr)
        raise SystemExit(1)
