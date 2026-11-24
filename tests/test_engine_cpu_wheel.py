import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import zipfile


def test_cpu_variant_preserves_native_code_and_has_valid_records(tmp_path):
    derive = runpy.run_path(str(Path(__file__).parents[1] / "scripts/engine_cpu_wheel.py"))[
        "derive"
    ]
    source = tmp_path / "kumo_engine-1.0.1-cp311-abi3-manylinux_2_34_x86_64.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("kumo/engine/example.py", "version = 'upstream'\n")
        archive.writestr("diskgraph/native.so", b"original native bytes")
        executable = zipfile.ZipInfo("diskgraph/bin/ingest")
        executable.external_attr = 0o100755 << 16
        archive.writestr(executable, b"original executable bytes")
        archive.writestr(
            "kumo_engine-1.0.1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: kumo-engine\nVersion: 1.0.1\nRequires-Dist: faiss-gpu==1.14.3\n",
        )
        archive.writestr("kumo_engine-1.0.1.dist-info/RECORD", "")
    destination = derive(source, tmp_path / "cpu")
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("diskgraph/native.so") == b"original native bytes"
        assert archive.read("diskgraph/bin/ingest") == b"original executable bytes"
        assert archive.getinfo("diskgraph/bin/ingest").external_attr >> 16 == 0o100755
        assert archive.read("kumo/engine/example.py") == b"version = 'upstream'\n"
        meta = archive.read("kumo_engine-1.0.1+cpu.dist-info/METADATA").decode()
        assert (
            "Version: 1.0.1+cpu" in meta and "faiss-cpu==1.14.3" in meta and "faiss-gpu" not in meta
        )
        notice = json.loads(archive.read("kumo_engine-1.0.1+cpu.dist-info/CPU_VARIANT.json"))
        assert notice["upstream_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        record = archive.read("kumo_engine-1.0.1+cpu.dist-info/RECORD").decode()
        for name, digest, size in csv.reader(io.StringIO(record)):
            if digest:
                data = archive.read(name)
                expected = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                )
                assert digest == "sha256=" + expected and int(size) == len(data)
