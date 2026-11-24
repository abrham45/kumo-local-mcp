# kumo-local-mcp

A local stdio MCP server for NVIDIA Kumo models. Your PC runs inference and training; datasets,
weights, checkpoints, and predictions stay on your machine. No UI, hosted service, API key, or cloud
account is required for foundation-model inference. An MCP client can read permitted results.

This is an independent project, not an NVIDIA product. Source and releases are maintained at
[abrham45/kumo-local-mcp](https://github.com/abrham45/kumo-local-mcp).

## Install and configure

Use Python 3.11 or later. Python 3.12 is a reasonable starting point for a new installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install git+https://github.com/abrham45/kumo-local-mcp.git
mkdir -p data
kumo-local-mcp --config ./config.json setup \
  --allow-input ./data \
  --output ./kumo-output \
  --backend tabular --backend relational --backend forecast --backend anomaly \
  --device cpu --install
```

For a published PyPI release, use `python -m pip install kumo-local-mcp` instead. To develop from a
checkout, use `python -m pip install -e '.[dev]'`. See
[CONTRIBUTING.md](https://github.com/abrham45/kumo-local-mcp/blob/main/CONTRIBUTING.md).

`setup` creates an SDM worker environment beside the configuration file and checks imports and
accelerators. Add `--backend engine` to install a separate Engine environment. Installation progress,
downloads, and failures appear locally. Failed installs remain unavailable in discovery. No tool
installs packages or runs an assistant-supplied shell command.

The adapter targets `structured-data-models==0.3.0`, whose public model API includes all four
families. An older installation may lack Forecast and Anomaly; the server reports that limitation.
Legacy Kumo-TS `sdk` environments are not supported by this release.

Weights download on first model use **only if enabled**:

```bash
kumo-local-mcp --config ./config.json setup --allow-downloads
```

Otherwise the worker sets `HF_HUB_OFFLINE=1` and uses the local cache at
`<output>/.kumo/weights`. A missing checkpoint produces a failed job with an action to check local
weights and configuration. Model caches may take multiple GB; large models need substantial memory.
SDM auto-device selection uses CUDA when available and CPU otherwise. Explicit SDM MPS execution is
currently unsupported. Engine device selection depends on its installed public API.

To prepare weights once while keeping server execution offline, run:

```bash
kumo-local-mcp --config ./config.json setup --download-models --download-workers 8
```

This downloads pinned public model revisions, resumes interrupted transfers, and verifies SHA-256
before installing weights in the Hugging Face cache. It does not enable automatic server downloads.
`model_cache_directory` in the local configuration can select an existing cache.

### Engine on macOS

Engine 1.0.1 supplies a Linux x86-64 native wheel. The optional Docker runtime lets the Mac server
run Engine locally in a Linux CPU container:

```bash
docker build --platform linux/amd64 -f Dockerfile.engine -t kumo-local-engine:0.1.0-cpu .
kumo-local-mcp --config ./config.json setup \
  --backend tabular --backend relational --backend forecast --backend anomaly --backend engine \
  --engine-docker-image kumo-local-engine:0.1.0-cpu --device auto
```

The build creates a local `kumo-engine==1.0.1+cpu` variant: upstream Python/native code is unchanged;
the dependency switches from GPU Faiss to CPU Faiss. A provenance record stores the upstream wheel
hash. This derived wheel is for local use and is never included in this project's distributions.
The image build requires network access. Runtime containers have networking disabled, read-only
system files, and mounts limited to the worker package and configured output directory. They use
CPU PyTorch and may run more slowly under x86 emulation. Docker must be running; serving never
pulls or builds an image automatically. Native Linux Engine environments remain supported.
The container's 64 MiB shared-memory limit selects Engine's inline data loader, avoiding the
pidfd-based multiprocessing path unsupported by this Mac's x86 emulation.

Start the server:

```bash
kumo-local-mcp --config ./config.json serve
```

Stdout contains only MCP messages. Application and worker logs go to stderr. One server owns an output
registry at a time. The configuration path can also be set with `KUMO_LOCAL_MCP_CONFIG`.

## Connect an MCP client

Use absolute paths in your client's MCP configuration:

```json
{
  "mcpServers": {
    "kumo-local": {
      "command": "/absolute/path/to/.venv/bin/kumo-local-mcp",
      "args": ["--config", "/absolute/path/to/config.json", "serve"]
    }
  }
}
```

## Privacy and file access

Default disclosure settings deny column names, summaries, and sample values. Enable each separately
with `setup --allow-columns`, `--allow-summaries`, or `--allow-samples`, or edit the JSON locally and
restart. Samples are bounded; when names are disabled, sample keys are anonymized. The same policy
filters tools and both MCP resource templates. Resource reads can disclose data to an assistant.

Input paths resolve inside allowed roots; sibling prefixes, traversal, and symlink escapes are
rejected. Registration copies a bounded local CSV or Parquet snapshot, records SHA-256 and schema,
and checks the snapshot before subsequent use. Later source edits create new dataset versions.
Parquet requires `pyarrow` in the server environment (`pip install pyarrow`). CSV works in the base
installation. Database and remote connector registration are unsupported in this release; database
credentials are never requested or returned.

Exports must use a new filename under the output directory. Never point an output directory at
another application's files. Internal raw metadata, generated training data, model files, and reports
remain local private files, not general-purpose MCP file resources. Treat registered contents and
previews as untrusted data. Workers are dependency-isolation processes, not an OS security sandbox
against other software running as your user.

## Tools and workflow

1. Call `list_capabilities`. It imports each enabled backend in a worker and reports versions,
   devices, operations, experimental adapters, and limitations without loading weights.
2. Call `register_dataset` with a permitted CSV/Parquet path. Retain `dataset_id`; inspect or preview
   it only as disclosure permits.
3. Submit a prediction or evaluation tool. It returns `{"status":"queued","job_id":"job_…"}`.
4. Poll `get_job_status` until its state is `succeeded`, `failed`, or `cancelled`.
   Progress is phase-based (queued, executing, finished), not a fabricated percentage of model work.
5. Use `result_id` with `get_result_summary`, `preview_result`, or `export_result`.

| Area | Tools |
|---|---|
| Discovery/data | `list_capabilities`, `register_dataset`, `inspect_dataset`, `preview_dataset` |
| Schema | `create_relational_schema`, `validate_relational_schema` |
| Prediction | `predict_tabular`, `predict_relational`, `forecast_timeseries`, `detect_anomalies` |
| Evaluation | `evaluate_tabular`, `evaluate_relational`, `backtest_forecast`, `evaluate_anomalies` |
| Sessions/jobs | `create_session`, `list_sessions`, `close_session`, `get_job_status`, `get_job`, `cancel_job`, `list_objects` |
| Results | `get_result_summary`, `preview_result`, `export_result`, `explain_result` |
| Adaptation | `start_finetuning` |
| Engine | `engine_create_graph`, `engine_create_snapshot`, `engine_generate_training_data`, `engine_train_model`, `engine_predict` |

MCP resources: `kumo://results/{result_id}` and `kumo://reports/{result_id}` contain disclosure-filtered
reports. Raw `metadata.json` includes execution inputs for reproducibility and is never exposed as an
MCP resource. Error responses contain a code, readable message, and suggested next action; raw
backend exception details remain in local stderr.

### Model support

| Family | Prediction | Evaluation | Explanation | Fine-tuning |
|---|---|---|---|---|
| Tabular | Classification probabilities; 999 regression quantiles | Held-out accuracy, F1, log loss, MAE, RMSE | Absolute gradients of preprocessed numeric query inputs | Disjoint context/query batches, validation checkpoint selection |
| Relational | Connected-table task rows and keys; experimental adapter | Held-out task rows; related tables exclude target columns | Absolute gradients when the installed SDM API supports them | Experimental differentiable training |
| Forecast | Regular series, ≥512 historical rows, multiple targets, past covariates, DARR settings | Rolling-origin MAE/RMSE with history-only preprocessing | Not currently exposed | Experimental chronological windows and held-out MSE selection |
| Anomaly | MAE reconstruction scores, SCS/MACS labels, optional explicit MAE threshold | Precision/recall/F1 with binary labels; diagnostics without labels | Not currently exposed | Experimental reconstruction-loss adaptation on nominal windows |
| Engine | Local graph/snapshot, PQL training-data generation, training, checkpoint prediction | Backend training metrics | Not currently exposed | Task-specific graph model training |

Unsupported advanced operations fail explicitly; no substitute model or hosted API is used. The
upstream SDM project documents broader differentiable training possibilities; this package exposes
only the adapter paths above. Forecast and Anomaly fine-tuning remain experimental; a backend that
does not preserve objective gradients is rejected. Actual synthetic workflow results are recorded in
`VALIDATION.md`.
Gradient explanations describe sensitivity of **preprocessed numerical inputs**, not causal feature
importance. Exported gradient columns use positional IDs because preprocessing may transform columns.

Native SDM Relational consumes labeled task datasets plus a schema/task link. It does not execute PQL.
Use Engine for predictive queries and point-in-time graph sampling. The SDM relational adapter rejects
declared temporal tables because it does not implement per-prediction temporal graph filtering.

### Example tabular request

After registering context and query datasets:

```json
{
  "request": {
    "context_dataset_id": "dataset_…",
    "query_dataset_id": "dataset_…",
    "target": "outcome",
    "task": "classification",
    "backend_options": {"size": "small", "num_estimators": 1, "seed": 0}
  }
}
```

The context includes labels; the query may omit them. Feature names and order must match. Regression
returns native quantiles, with `q500` as the median. Advanced options use a strict typed schema;
unknown fields, arbitrary paths, Python callbacks, and executable code are rejected.

Evaluation requires a separate labeled query dataset and `split: {"method":"holdout"}` or a valid
temporal split with `time_column`. Overlapping rows are rejected. Forecast backtests take
`initial_history`, `folds`, and `stride`; every validation window follows its own historical cutoff.
Anomaly labels must be binary and excluded from input features. An explicit anomaly threshold is in
the backend's MAE score units, not a calibrated probability.

### Fine-tuning

`start_finetuning` accepts a family-specific typed request. Tabular and Relational specify target,
task, disjoint training/validation datasets, context/query batch sizes, steps, and learning rate;
Relational also requires `schema_id`. Forecast specifies timestamp, targets, covariates, horizon,
training/validation IDs, steps, and learning rate. Each split needs at least 512 rows plus the horizon,
and validation timestamps must follow training timestamps. Forecast defaults to `trainable_scope:
"head"`, adapting its head/cross-channel layers with its encoder frozen. `"all"` also trains the
encoder and needs substantially more memory. Anomaly specifies numeric columns,
training/validation IDs, window size, steps, and learning rate. Its adaptation objective assumes
nominal data and does not estimate anomaly accuracy. All paths retain the best validation checkpoint,
including the pretrained baseline, and return a checkpoint ID through the completed result.

Engine training accepts `max_epochs`, `num_experiments`, `max_steps_per_epoch`, `max_eval_steps`,
and `refit_trainval`. Defaults bound the search to one experiment and 100 steps per phase, with no
refit. These limits control work; use representative validation data before interpreting metrics.

### Sessions and jobs

`create_session` prepares a Tabular context or loads a Forecast/Anomaly model in its own worker.
Reuse its ID with matching settings. Idle workers unload, and the next use recreates model/context.
Sessions expire after server restart; dataset, schema, result, checkpoint, and job records persist in
SQLite. Interrupted jobs are publicly reported as `failed` with an interruption error; they are not
silently restarted. `get_job` remains an alias for `get_job_status`. Cancellation and execution
timeouts terminate worker process groups; any retained incomplete files are identified as partial
artifacts rather than valid results. Other session workers remain available. Heavy concurrency
defaults to one; the queue, execution timeout, input size, preview size, and idle TTL are configurable.

## Verify and build

```bash
python -m pytest -q
ruff check .
python -m build
python -m twine check dist/*
```

Real model tests deliberately require a separate invocation and do not substitute fake predictions:

```bash
python -m pip install -e '.[sdm]'
python examples/smoke_models.py --allow-downloads --finetune --explain
# On a supported Engine installation:
python examples/smoke_models.py --families engine
# Comprehensive offline workflows after setup --download-models:
python examples/verify_workflows.py --families tabular relational forecast anomaly \
  --weights-cache ./kumo-output/.kumo/weights --workdir /tmp/kumo-verification
# Include --families engine --engine-docker-image kumo-local-engine:0.1.0-cpu for Engine.
```

The smoke runner starts a real stdio client, registers synthetic local data, invokes actual Kumo
models, and writes `smoke-report.json`. Its exit status is nonzero when any requested family fails.
Small synthetic examples verify integration, not production model quality. See `VALIDATION.md` for
observed results and remaining release gates, `docs/ARCHITECTURE.md` for the implementation design,
and `docs/RELEASING.md` for publication.

## Upstream references

- [SDM release and installation](https://pypi.org/project/structured-data-models/)
- [SDM model APIs](https://nvidia.github.io/structured-data-models/api/models.html)
- [SDM fine-tuning](https://nvidia.github.io/structured-data-models/finetuning.html)
- [Engine local SDK](https://github.com/NVIDIA/kumo-engine/tree/main/packages/kumo-engine)
- [Engine model plan](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/train-a-model/model-plan-guide)

Project source license: MIT. Model weights and upstream packages have their own licenses. Model
weights are downloaded separately and are not bundled in this distribution.
