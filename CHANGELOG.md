# Changelog

## 0.1.0

- Added explicit source archive inputs, package metadata, contributor instructions, and ignore rules
  for local configuration, secrets, environments, generated data, model artifacts, and tool caches.
- Added local stdio MCP tools, configuration and setup, privacy-controlled dataset registration,
  persistent job/results registry, background workers, cancellation, sessions, and exports.
- Added SDM adapters for Tabular, Relational, Forecast and Anomaly, held-out evaluation, gradient
  explanations for supported models, and validated SDM fine-tuning/checkpoint paths.
- Added experimental local Engine graph, snapshot, training-data, training, and prediction tools.
- Added boundary, recovery, worker-failure, cancellation, timeout, and stdio tests plus a real-model
  smoke runner, architecture documentation, and release packaging instructions.
- Added verified resumable weight preparation, canonical job-status and session-list tools,
  comprehensive real MCP workflow verification, and an isolated local Linux CPU Engine runtime.
- Verified real inference, evaluation, supported gradients, one-step adaptation, checkpoint reloads,
  and reusable sessions across all four SDM families.
- Verified the complete local Engine graph/snapshot/PQL/training/checkpoint-prediction workflow,
  preserving executable permissions and using inline data loading for Mac x86 emulation.
