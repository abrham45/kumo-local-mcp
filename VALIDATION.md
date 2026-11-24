# Local validation record — November 24, 2026

This record describes the local verification of the initial `kumo-local-mcp` release candidate.
Publication status is recorded by the GitHub release and PyPI project; local verification alone
does not establish that a version was published.

## Real SDM workflows

The comprehensive verifier uses an actual MCP stdio client/server, isolated model workers,
synthetic CSV datasets, and real NVIDIA Kumo weights. Server runs are offline. All six weight
files were downloaded from pinned public revisions and SHA-256 verified before use.

| Family | Successful checks | Covered behavior |
|---|---:|---|
| Tabular | 14 | Classification and regression, held-out metrics, finite saved predictions, CSV/Parquet/JSON exports, resource parity, numerical gradients, one-step adaptation, checkpoint inference, reusable sessions |
| Relational | 12 | Classification and regression over connected tables, metrics, saved outputs/exports, task and related-table gradients, one-step adaptation, checkpoint inference |
| Forecast | 9 | Prediction, rolling-origin backtest, exports, multiple targets with integer covariates, finite outputs, missing-history interpolation, session reuse, head/cross-channel adaptation, checkpoint inference |
| Anomaly | 7 | Numeric and integer sensors, labeled evaluation, unlabeled diagnostics, saved outputs/exports, session reuse, nominal-window adaptation, checkpoint inference |

**42 real workflow checks passed.** Forecast checks include two output-validation checks; the
counts describe verifier assertions, not independent model benchmarks. Adaptation uses one gradient
step and selects the best held-out checkpoint, including the baseline. These small synthetic runs
prove integration; they do not establish production accuracy or that every adaptation improves it.
Forecast and Anomaly adaptation, native relational adaptation, and the Engine adapter remain marked
experimental. Full-encoder Forecast training, GPU/MPS execution, and every backend option have not
been exercised on this machine.

Evidence is retained under `validation/supervised` and `validation/timeseries`.
The verifier is `examples/verify_workflows.py`; each report includes actual outcomes, timing, result
IDs, capabilities, and artifact locations. Generated data and checkpoints are local runtime artifacts
and are excluded from distribution archives.

## Boundary and lifecycle verification

Unit/stdio coverage includes immutable dataset versions and tampering, path/symlink escapes,
input limits, duplicate headers, schema integrity and Engine primary-key links, validation overlap,
privacy defaults, filtered MCP resources, secret exclusion, typed options, model-cache recovery after
training failure, crashed workers, running/queued cancellation, partial-output identification,
timeouts, restart recovery, canonical job states, and session listing.

Download tests use a local HTTP endpoint to verify SHA-256, corrupt-content rejection, restart
resumption, partial-byte response recovery, and expired-link refresh. The CPU Engine wheel test
verifies unchanged upstream code/native bytes, preserved executable permissions, provenance, and
every wheel RECORD hash.

Latest source checks: 31 unit/stdio tests passed; Ruff lint and formatting passed. A built wheel was
installed in a fresh environment using local distributions, passed `pip check`, and passed all 31
tests without importing the source tree. Final installed-package checks are recorded in
`validation/package-checks.json`. The archives' metadata and contents are inspected before delivery;
they exclude configurations, registries, environments, generated data, and weights.

## Engine runtime

The native Engine 1.0.1 wheel targets Linux x86-64. This Mac uses a separate Linux/amd64 Docker CPU
runtime. The local `1.0.1+cpu` dependency variant preserves upstream code/native binaries and replaces
GPU Faiss with CPU Faiss; its provenance records the original wheel hash. It is not published or
included in this package. Runtime containers have no network and mount only their package and output
root. **Six real Engine checks passed**: graph validation, snapshot ingestion (7,320 nodes and 14,400
edges), PQL training-data generation, one-epoch/two-step training, saved-bundle prediction, and
finite predictions with CSV/Parquet/JSON exports and MCP resource parity. Evidence lives in
`validation/engine/verification.json`. Together with SDM, **48 real MCP workflow checks passed**.

Two additional real-container lifecycle checks passed: runtime restrictions and successful probe/
cleanup, and cancellation during backend import with container/CLI cleanup. The runtime uses 64 MiB
of shared memory so Engine selects inline data loading; its multiprocessing pidfd path is unsupported
by this Mac's x86 emulation. No upstream model code was patched. The tested Linux runtime uses
Python 3.12.16, CPU PyTorch 2.14.1, Engine 1.0.1+cpu, CPU Faiss 1.14.3 and PyArrow 24.0.0.
`validation/engine-runtime-manifest.json` records all 73 package versions and wheel hashes.
An additional installed-package run restarted the MCP server and Engine worker, reused the persisted
snapshot/checkpoint, and reproduced predictions when compared by entity. Engine output row ordering
is not guaranteed; retain its `ENTITY` column when joining predictions.

## Environment and limits

Mac server/model runtime: Python 3.13.13, MCP SDK 1.27.0, pandas 3.0.2, Pydantic 2.13.2,
SDM 0.3.0, PyTorch 2.10.0, PyArrow 23.0.1, scikit-learn 1.7.1. Models used CPU, small Tabular
variants, default Relational, Forecast and Anomaly checkpoints. All data was synthetic and stayed
local. Public package/model downloads required no account or API token.

The saved production configuration denies column names, summaries, and samples and disables automatic
model downloads. Synthetic verifier configurations enable disclosures only for their own generated
data. A local test does not verify the complete Python/OS matrix; the CI definition remains unrun on
a remote service at the time of these local checks. GitHub Actions performs CI and release checks.

## Start the installed server

```bash
cd /path/to/kumo-local-mcp
.venv/bin/kumo-local-mcp --config config.json serve
```
