# Contributing

Use Python 3.11 or later. Keep the MCP server independent of heavyweight model imports; run model
operations in workers and preserve stdout for protocol messages. All tools and MCP resources must
respect the same disclosure settings and configured file roots.

## Development checks

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python -m build
python -m twine check --strict dist/*
```

The normal suite uses local fixtures and a loopback HTTP server; it does not download model weights.
For a model adapter change, also run the affected real workflows in `examples/verify_workflows.py`
with a prepared weight cache. Record the actual backend version and limitations. Synthetic workflows
verify integration, not production model quality. Engine's Docker runtime needs a separate local
image; see README. Never commit generated predictions, checkpoints, model weights, or local reports.

## Preparing a change

Keep changes focused, document user-visible behavior, and include checks that catch meaningful
regressions. Do not add credentials, local configuration, or private datasets. Only the two curated
CSV files and sanitized configuration under `examples` belong in the distribution. Inspect both
`git status --short` and archive contents before a release; `.gitignore` does not untrack files
already committed. Follow `docs/RELEASING.md` for the release procedure. Pushes and publication require
the owner's instruction.
