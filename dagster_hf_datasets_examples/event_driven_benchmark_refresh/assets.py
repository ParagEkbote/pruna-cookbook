from __future__ import annotations

import json
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    DefaultSensorStatus,
    MaterializeResult,
    RunRequest,
    SensorEvaluationContext,
    asset,
    sensor,
)
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset, load_dataset_builder

# ── Helpers ───────────────────────────────────────────────────────────────────

_STATE_FILE = Path(".dagster_hf_storage/sensor_state/humaneval_last_seen.json")


def _get_current_hub_revision(path: str, config: str | None = None) -> str:
    """Fetch the latest dataset revision (commit SHA) from the Hub.

    Uses load_dataset_builder which performs a lightweight HEAD-equivalent
    request against the Hub API without downloading the dataset itself.
    Returns the dataset_info hash as a stable revision identifier.
    """
    builder = load_dataset_builder(path, config)
    # Use the builder's dataset_info hash as a proxy for revision.
    # In production, swap this for huggingface_hub.DatasetCard.load()
    # and inspect the card's last_modified timestamp for a real SHA.
    info = builder.info
    return str(hash(str(info.description) + str(info.version)))


def _read_last_seen_revision() -> str | None:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text()).get("revision")
    except Exception:
        pass
    return None


def _write_last_seen_revision(revision: str) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"revision": revision}))


# ── Assets ────────────────────────────────────────────────────────────────────

@hf_dataset_asset(
    path="openai/openai_humaneval",
    split="test",
    group_name="event_driven_refresh",
    io_manager_key="hf_parquet_io_manager",
)
def humaneval_benchmark(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Materialize the HumanEval benchmark evaluation set.

    HumanEval contains 164 Python programming challenges used to
    evaluate LLM code generation. This asset is the target of the
    sensor — it re-materializes whenever the Hub dataset revision
    changes, ensuring downstream evaluation always runs against
    the canonical upstream version.
    """
    context.log.info("Loaded HumanEval: %s tasks", len(dataset))

    # Characterize the benchmark
    prompt_lengths = [len(ex["prompt"].split()) for ex in dataset]
    avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths)
    has_canonical_solution = sum(1 for ex in dataset if ex.get("canonical_solution"))

    return MaterializeResult(
        value=dataset,
        metadata={
            "tasks": len(dataset),
            "columns": dataset.column_names,
            "avg_prompt_tokens": round(avg_prompt_len, 1),
            "tasks_with_canonical_solution": has_canonical_solution,
            "source_dataset": "openai/openai_humaneval",
            "split": "test",
            "fingerprint": dataset._fingerprint,
        },
    )


@asset(
    group_name="event_driven_refresh",
    io_manager_key="hf_parquet_io_manager",
)
def humaneval_formatted(
    context,
    humaneval_benchmark: Dataset,
) -> MaterializeResult:
    """Reformat HumanEval tasks into a flat evaluation-ready schema.

    Merges `prompt` + `canonical_solution` into a single `reference`
    field and normalises the `task_id` column. This is the asset
    consumed by downstream evaluation harnesses — it re-runs
    automatically whenever `humaneval_benchmark` is refreshed.
    """
    formatted = humaneval_benchmark.map(
        lambda ex: {
            "task_id": ex["task_id"],
            "prompt": ex["prompt"].strip(),
            "reference": (ex["prompt"] + ex["canonical_solution"]).strip(),
            "test": ex["test"].strip(),
            "entry_point": ex["entry_point"],
        },
        remove_columns=humaneval_benchmark.column_names,
        desc="Formatting HumanEval tasks",
    )

    context.log.info("Formatted %s tasks", len(formatted))
    context.add_output_metadata(
        {
            "tasks": len(formatted),
            "output_columns": formatted.column_names,
        }
    )
    return MaterializeResult(
            value=formatted,
            metadata={
                "tasks": len(formatted),
                "output_columns": formatted.column_names,
            },
        )


@asset(
    group_name="event_driven_refresh",
)
def benchmark_refresh_report(
    context,
    humaneval_benchmark: Dataset,
    humaneval_formatted: Dataset,
) -> MaterializeResult:
    """Emit a report confirming the refresh cycle completed successfully.

    Records the fingerprint of the refreshed dataset and the run ID
    that triggered the refresh, creating an audit trail of when and
    why the benchmark was re-materialized.
    """
    report = {
        "raw_tasks": len(humaneval_benchmark),
        "formatted_tasks": len(humaneval_formatted),
        "raw_fingerprint": humaneval_benchmark._fingerprint,
        "formatted_columns": humaneval_formatted.column_names,
        "dagster_run_id": context.run_id,
    }

    context.log.info("Refresh cycle complete: %s", report)
    context.add_output_metadata(report)
    return MaterializeResult(
        value=report,
        metadata=report,
    )


# ── Sensor ────────────────────────────────────────────────────────────────────
#
# The sensor polls the Hub for a new revision of openai/openai_humaneval
# on each evaluation tick (default: every 30 seconds in dev, configure
# minimum_interval_seconds for production).
#
# When a new revision is detected:
#   1. The sensor emits a RunRequest targeting humaneval_benchmark
#   2. Dagster materializes humaneval_benchmark
#   3. humaneval_formatted and benchmark_refresh_report re-run downstream
#
# State is persisted to disk in _STATE_FILE. In production, use
# context.cursor (SensorEvaluationContext) instead of a file for
# distributed/multi-replica deployments.

@sensor(
    asset_selection=[
        AssetKey("humaneval_benchmark"),
        AssetKey("humaneval_formatted"),
        AssetKey("benchmark_refresh_report"),
    ],
    default_status=DefaultSensorStatus.STOPPED,  # flip to RUNNING in production
    minimum_interval_seconds=300,                 # poll every 5 minutes
    description="Re-materializes HumanEval assets when the Hub dataset revision changes.",
)
def humaneval_revision_sensor(context: SensorEvaluationContext):
    """Poll the HumanEval Hub dataset for revision changes.

    On each tick, fetches the current revision identifier from the Hub.
    If it differs from the last seen revision stored in cursor state,
    emits a RunRequest to re-materialize the full benchmark refresh pipeline.

    Uses context.cursor for state persistence — safe across restarts
    and compatible with multi-replica Dagster deployments.
    """
    last_seen = context.cursor or ""

    try:
        current_revision = _get_current_hub_revision("openai/openai_humaneval")
    except Exception as e:
        context.log.warning("Hub revision check failed: %s", e)
        return  # Skip tick silently — don't crash the sensor

    if current_revision == last_seen:
        context.log.info("No revision change detected (revision=%s). Skipping.", current_revision)
        return

    context.log.info(
        "Revision change detected: %s → %s. Triggering refresh.",
        last_seen or "<none>",
        current_revision,
    )

    # Persist new revision to cursor before emitting run
    context.update_cursor(current_revision)

    yield RunRequest(
        run_key=f"humaneval_refresh_{current_revision}",
        run_config={},
        tags={
            "trigger": "hub_revision_change",
            "previous_revision": last_seen or "initial",
            "current_revision": current_revision,
        },
    )
