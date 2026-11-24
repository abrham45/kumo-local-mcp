"""Derive a local CPU-only Engine wheel; preserve all upstream Python/native code bytes."""

from __future__ import annotations

import argparse
import base64
import csv
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile


def derive(source: Path, directory: Path) -> Path:
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source) as archive:
        old_metadata = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        old_root = old_metadata.rsplit("/", 1)[0]
        text = archive.read(old_metadata).decode()
        version_line = next(line for line in text.splitlines() if line.startswith("Version: "))
        version = version_line.removeprefix("Version: ")
        if "+" in version or "Requires-Dist: faiss-gpu" not in text:
            raise ValueError("Expected an unmodified Engine wheel with the GPU Faiss requirement")
        new_version = version + "+cpu"
        new_root = old_root.replace(version, new_version)
        text = text.replace(version_line, "Version: " + new_version, 1).replace(
            "Requires-Dist: faiss-gpu", "Requires-Dist: faiss-cpu"
        )
        members, attributes = {}, {}
        for name in archive.namelist():
            if name.endswith("/") or name == old_root + "/RECORD":
                continue
            new_name = (
                name.replace(old_root + "/", new_root + "/", 1)
                if name.startswith(old_root + "/")
                else name
            )
            members[new_name] = text.encode() if name == old_metadata else archive.read(name)
            attributes[new_name] = copy.copy(archive.getinfo(name))
            attributes[new_name].filename = new_name
        for name in archive.namelist():
            if name.endswith((".py", ".so")):
                assert members[name] == archive.read(name)
    notice = {
        "upstream_wheel": source.name,
        "upstream_sha256": original_digest,
        "changes": ["Local version suffix +cpu", "faiss-gpu dependency replaced with faiss-cpu"],
        "python_and_native_code": "byte-for-byte unchanged",
        "publication": "local use only",
    }
    members[new_root + "/CPU_VARIANT.json"] = json.dumps(notice, indent=2).encode()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([new_root + "/RECORD", "", ""])
    members[new_root + "/RECORD"] = record.getvalue().encode()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name.replace("-" + version + "-", "-" + new_version + "-", 1)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as output:
        for name, data in members.items():
            output.writestr(attributes.get(name, name), data)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(derive(args.wheel, args.output))
