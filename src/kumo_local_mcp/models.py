from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .core import StrictModel

Family = Literal["tabular", "relational", "forecast", "anomaly", "engine"]
Task = Literal["classification", "regression"]


class BackendOptions(StrictModel):
    size: Literal["small", "medium", "large"] = "small"
    num_estimators: int = Field(default=1, ge=1, le=32)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    cross_channel: bool = True
    alpha: float = Field(default=1.0, ge=0, le=1)
    k: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.05, gt=0, le=10)
    num_samples: int = Field(default=30, ge=1, le=100)
    num_steps: int = Field(default=20, ge=1, le=100)


class TableSpec(StrictModel):
    dataset_id: str
    primary_key: str | None = None
    time_column: str | None = None


class Relationship(StrictModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class TaskLink(StrictModel):
    task_column: str
    table: str
    table_column: str


class TabularRequest(StrictModel):
    context_dataset_id: str
    query_dataset_id: str
    target: str
    task: Task
    session_id: str | None = None
    checkpoint_id: str | None = None
    backend_options: BackendOptions = Field(default_factory=BackendOptions)


class RelationalRequest(TabularRequest):
    schema_id: str
    # Native SDM uses labeled task rows and key relationships; PQL is an Engine operation.
    predictive_query: str | None = None


class ForecastRequest(StrictModel):
    dataset_id: str
    timestamp: str
    targets: list[str] = Field(min_length=1, max_length=40)
    horizon: int = Field(ge=1, le=4096)
    covariates: list[str] = []
    missing_values: Literal["backend", "reject", "interpolate"] = "backend"
    session_id: str | None = None
    checkpoint_id: str | None = None
    backend_options: BackendOptions = Field(default_factory=BackendOptions)

    @model_validator(mode="after")
    def unique_columns(self):
        cols = self.targets + self.covariates + [self.timestamp]
        if len(cols) != len(set(cols)):
            raise ValueError("Target, covariate, and timestamp columns must be distinct.")
        return self


class AnomalyRequest(StrictModel):
    dataset_id: str
    columns: list[str] = Field(min_length=1, max_length=40)
    threshold: float | None = Field(default=None, ge=0)
    label_method: Literal["scs", "macs"] = "macs"
    session_id: str | None = None
    checkpoint_id: str | None = None
    backend_options: BackendOptions = Field(default_factory=BackendOptions)


class EvaluationSplit(StrictModel):
    method: Literal["holdout", "temporal"]
    # Evaluation always receives separate context and validation datasets.
    time_column: str | None = None


class FineTuneRequest(StrictModel):
    family: Literal["tabular", "relational"]
    dataset_id: str
    validation_dataset_id: str
    target: str
    task: Task
    schema_id: str | None = None
    steps: int = Field(default=30, ge=1, le=10000)
    context_rows: int = Field(default=64, ge=2, le=8192)
    query_rows: int = Field(default=32, ge=1, le=4096)
    learning_rate: float = Field(default=1e-5, gt=0, le=0.01)
    backend_options: BackendOptions = Field(default_factory=BackendOptions)


class ForecastFineTuneRequest(ForecastRequest):
    family: Literal["forecast"] = "forecast"
    validation_dataset_id: str
    steps: int = Field(default=10, ge=1, le=10000)
    learning_rate: float = Field(default=1e-5, gt=0, le=0.01)
    trainable_scope: Literal["head", "all"] = "head"


class AnomalyFineTuneRequest(AnomalyRequest):
    family: Literal["anomaly"] = "anomaly"
    validation_dataset_id: str
    steps: int = Field(default=10, ge=1, le=10000)
    window_rows: int = Field(default=128, ge=8, le=512)
    learning_rate: float = Field(default=1e-5, gt=0, le=0.01)


class EngineQuery(StrictModel):
    snapshot_id: str
    predictive_query: str = Field(min_length=1, max_length=10000)


class EngineTrain(EngineQuery):
    # Small, audited public training knobs only; no connector paths or arbitrary objects.
    max_epochs: int = Field(default=5, ge=1, le=100)
    num_experiments: int = Field(default=1, ge=1, le=20)
    max_steps_per_epoch: int = Field(default=100, ge=1, le=4000)
    max_eval_steps: int = Field(default=100, ge=1, le=8000)
    refit_trainval: bool = False


class EnginePredict(EngineQuery):
    checkpoint_id: str
    threshold: float = Field(default=0.5, ge=0, le=1)
