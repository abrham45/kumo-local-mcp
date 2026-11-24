from __future__ import annotations

import json
import csv
from pathlib import Path

import pandas as pd

from .core import Config, Failure, Store, fingerprint, uid, within


def read_frame(path: Path, max_rows: int):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if (
            not header
            or len(header) != len(set(header))
            or any(not name.strip() for name in header)
        ):
            raise Failure("DATA_INVALID", "CSV column names must be nonempty and unique.")
        frame = pd.read_csv(path, nrows=max_rows + 1)
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows > max_rows:
            raise Failure("DATA_LIMIT", "Dataset exceeds the configured row limit.")
        frame = pd.read_parquet(path)
    else:
        raise Failure("FORMAT_UNSUPPORTED", "Only CSV and Parquet files are supported.")
    if len(frame) > max_rows:
        raise Failure("DATA_LIMIT", "Dataset exceeds the configured row limit.")
    if frame.empty or not frame.columns.is_unique:
        raise Failure("DATA_INVALID", "Dataset must contain rows and unique column names.")
    frame.columns = frame.columns.astype(str)
    return frame


class DataManager:
    def __init__(self, config: Config, store: Store):
        self.config, self.store = config, store
        self.root = store.root / "datasets"
        self.root.mkdir(exist_ok=True)

    def register(self, source: str):
        path = within(Path(source), self.config.allowed_input_directories)
        if not path.is_file() or path.stat().st_size > self.config.max_input_bytes:
            raise Failure(
                "DATA_LIMIT", "Input must be a regular file within the configured size limit."
            )
        dataset_id = uid("dataset")
        destination = self.root / (dataset_id + path.suffix.lower())
        try:
            with path.open("rb") as src, destination.open("xb") as dst:
                remaining = self.config.max_input_bytes + 1
                while remaining:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
                if not remaining:
                    raise Failure("DATA_LIMIT", "Dataset grew beyond the configured size limit.")
            destination.chmod(0o600)
            frame = read_frame(destination, self.config.max_rows)
            return self.store.put(
                "dataset",
                {
                    "source": str(path),
                    "path": str(destination),
                    "sha256": fingerprint(destination),
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "dtypes": [str(v) for v in frame.dtypes],
                },
                dataset_id,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def descriptor(self, dataset_id: str):
        item = self.store.get(dataset_id, "dataset")
        path = within(Path(item["path"]), [self.root])
        if fingerprint(path) != item["sha256"]:
            raise Failure(
                "DATA_CHANGED",
                "A registered dataset snapshot changed.",
                "Register the source again to create a new dataset version.",
            )
        return item

    def frame(self, dataset_id: str):
        return read_frame(Path(self.descriptor(dataset_id)["path"]), self.config.max_rows)

    def inspect(self, dataset_id: str):
        item = self.descriptor(dataset_id)
        result = {
            "status": "completed",
            "dataset_id": item["id"],
            "fingerprint": item["sha256"],
            "warnings": [],
        }
        if self.config.disclosure.column_names:
            result["schema"] = dict(zip(item["columns"], item["dtypes"], strict=True))
        if self.config.disclosure.summaries:
            frame = self.frame(dataset_id)
            result["summary"] = {
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_values": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
            }
            if self.config.disclosure.column_names:
                result["missingness"] = {k: int(v) for k, v in frame.isna().sum().items()}
        return result

    def preview(self, frame, limit):
        if not self.config.disclosure.sample_values:
            raise Failure(
                "DISCLOSURE_DENIED",
                "Sample values are disabled in local configuration.",
                "Enable sample_values locally to let the assistant see data values.",
            )
        frame = frame.head(min(max(1, limit), self.config.max_preview_rows)).copy()
        if not self.config.disclosure.column_names:
            frame.columns = [f"column_{i + 1}" for i in range(len(frame.columns))]
        return {
            "status": "completed",
            "rows": json.loads(frame.to_json(orient="records", date_format="iso")),
            "warnings": [],
        }

    def validate_schema(self, tables, relationships, task_link=None):
        findings, frames = [], {}
        if not tables or len(tables) > 32:
            raise Failure("SCHEMA_INVALID", "Provide between 1 and 32 tables.")
        for name, table in tables.items():
            if not name.isidentifier() or name.startswith("_"):
                raise Failure("SCHEMA_INVALID", "Table aliases must be ordinary identifiers.")
            frame = self.frame(table["dataset_id"])
            frames[name] = frame
            key, stamp = table.get("primary_key"), table.get("time_column")
            if key and (
                key not in frame or frame[key].isna().any() or frame[key].duplicated().any()
            ):
                findings.append("A primary key is missing, null, or non-unique.")
            if stamp and (
                stamp not in frame or pd.to_datetime(frame[stamp], errors="coerce").isna().any()
            ):
                findings.append("A time column is missing or contains invalid timestamps.")
        for rel in relationships:
            left, right = frames.get(rel["from_table"]), frames.get(rel["to_table"])
            fk, pk = rel["from_column"], rel["to_column"]
            if left is None or right is None or fk not in left or pk not in right:
                findings.append("A relationship references an unknown table or column.")
            elif right[pk].isna().any() or right[pk].duplicated().any():
                findings.append("A relationship destination must be a unique non-null key.")
            elif not left[fk].dropna().isin(right[pk]).all():
                findings.append("A relationship contains orphan foreign keys.")
        if task_link:
            table = frames.get(task_link["table"])
            if table is None or task_link["table_column"] not in table:
                findings.append("The task link references an unknown table or key.")
        return findings

    def export(self, source: Path, name: str, format: str):
        if format not in {"csv", "parquet", "json"}:
            raise Failure("FORMAT_UNSUPPORTED", "Choose csv, parquet, or json.")
        target = within(
            self.config.output_directory / name, [self.config.output_directory], exists=False
        )
        if target.is_relative_to(self.store.root):
            raise Failure(
                "EXPORT_INVALID", "Exports cannot write into the internal state directory."
            )
        if (
            target == self.config.output_directory
            or target.exists()
            or target.suffix != "." + format
        ):
            raise Failure(
                "EXPORT_INVALID", "Choose a new output filename with the selected extension."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = read_frame(source, self.config.max_rows)
        with target.open("xb") as stream:
            try:
                if format == "csv":
                    stream.write(frame.to_csv(index=False).encode())
                elif format == "json":
                    stream.write(frame.to_json(orient="records", date_format="iso").encode())
                else:
                    frame.to_parquet(stream, index=False)
                target.chmod(0o600)
            except Exception:
                target.unlink(missing_ok=True)
                raise
        return {"status": "completed", "path": str(target), "warnings": []}
