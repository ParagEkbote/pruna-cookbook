# Event-Driven Benchmark Refresh

Automatically re-materialize evaluation datasets when an upstream Hub
benchmark changes, using a Dagster sensor with cursor-based state.

## What this example shows

- `@sensor` with `asset_selection` targeting multiple downstream assets
- Hub revision polling via `load_dataset_builder` (lightweight, no dataset download)
- `context.cursor` for durable sensor state across restarts and replicas
- `RunRequest` with `run_key` for idempotent trigger deduplication
- `DefaultSensorStatus.STOPPED` as a safe default — flip to `RUNNING` in production
- Downstream assets (`humaneval_formatted`, `benchmark_refresh_report`) re-running
  automatically when the ingestion asset is refreshed

## Dataset

[`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) — 164 Python programming
tasks used to benchmark LLM code generation capability. Compact and
stable, making revision changes meaningful when they do occur.

| Split | Rows | Description |
|-------|------|-------------|
| test | 164 | Python function completion tasks |

## Key API

```python
@sensor(
    asset_selection=[AssetKey("humaneval_benchmark"), ...],
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.STOPPED,
)
def humaneval_revision_sensor(context: SensorEvaluationContext):
    last_seen = context.cursor or ""
    current_revision = _get_current_hub_revision("openai/openai_humaneval")

    if current_revision == last_seen:
        return  # no change — skip this tick

    context.update_cursor(current_revision)
    yield RunRequest(run_key=f"humaneval_refresh_{current_revision}")
```

## Asset graph

```
[sensor tick]
      │  (revision changed)
      ▼
humaneval_benchmark         ← re-materialized on trigger
      │
      ▼
humaneval_formatted         ← auto-runs downstream
      │
      ▼
benchmark_refresh_report    ← audit record with run_id + fingerprint
```

## Sensor lifecycle

| Status | Behaviour |
|--------|-----------|
| `STOPPED` (default) | Sensor defined but not ticking — safe for dev |
| `RUNNING` | Polls Hub every `minimum_interval_seconds` |
| Revision unchanged | Tick completes, no run emitted |
| Revision changed | `RunRequest` emitted, full pipeline re-materializes |

## Cursor state

Sensor state is stored in `context.cursor` (Dagster-managed, persisted in
the instance storage). This is safe for multi-replica deployments.

The file-based fallback `_STATE_FILE` in `assets.py` is included only
as a local-dev reference — use `context.cursor` in production.

## Run deduplication

`run_key=f"humaneval_refresh_{current_revision}"` ensures that if Dagster
attempts to emit the same revision trigger twice (e.g. after a restart),
only one run is launched. Dagster deduplicates by `run_key` automatically.

## Hub revision detection

`load_dataset_builder()` fetches `DatasetInfo` without downloading data.
The revision proxy used here (`hash(description + version)`) is lightweight
but approximate. For production, use `huggingface_hub.list_dataset_commits()`
to get the real commit SHA:

```python
from huggingface_hub import list_dataset_commits
commits = list(list_dataset_commits("openai/openai_humaneval"))
latest_sha = commits[0].commit_id
```

## How to run

```bash
pip install dagster dagster-hf-datasets
dagster dev -f definitions.py
```

To test the sensor manually: in the Dagster UI go to **Automation → Sensors**,
find `humaneval_revision_sensor`, and click **Evaluate** to run one tick.
To enable continuous polling, toggle the sensor to **Running**.