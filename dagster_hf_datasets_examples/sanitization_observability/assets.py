import hashlib

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    MaterializeResult,
    asset,
    asset_check,
)
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


# ── Step 1: Ingest ────────────────────────────────────────────────────────────

@hf_dataset_asset(
    path="HuggingFaceFW/fineweb-edu",
    config="sample-10BT",
    split="train",
    group_name="sanitization_observability",
    io_manager_key="hf_parquet_io_manager",
)
def raw_fineweb_edu(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest the FineWeb-Edu 10BT sample from the Hugging Face Hub.

    This is a large web-crawl dataset with known quality variance —
    ideal for demonstrating realistic sanitization needs.
    """
    context.log.info("Ingested raw FineWeb-Edu: %s rows", len(dataset))

    null_text_count = sum(1 for ex in dataset if not ex.get("text"))
    short_text_count = sum(1 for ex in dataset if ex.get("text") and len(ex["text"].split()) < 10)

    context.log.info("Null/empty text rows: %s", null_text_count)
    context.log.info("Short text rows (<10 words): %s", short_text_count)

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "null_text_count": null_text_count,
            "short_text_count": short_text_count,
            "source_dataset": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "fingerprint": dataset._fingerprint,
        },
    )


# ── Step 2: Filter ────────────────────────────────────────────────────────────

@asset(
    group_name="sanitization_observability",
    io_manager_key="hf_parquet_io_manager",
)
def filtered_fineweb_edu(
    context: AssetExecutionContext,
    raw_fineweb_edu: Dataset,
) -> Dataset:
    """Remove null, empty, and very short text examples.

    Rows are dropped if:
    - `text` is None or empty string
    - `text` contains fewer than 10 whitespace-delimited tokens
    """
    before = len(raw_fineweb_edu)

    filtered = raw_fineweb_edu.filter(
        lambda ex: ex.get("text") is not None
        and len(ex["text"].strip()) > 0
        and len(ex["text"].split()) >= 10
    )

    after = len(filtered)
    dropped = before - after

    context.log.info("Filtered: %s rows → %s rows (%s dropped)", before, after, dropped)

    return filtered


# ── Step 3: Deduplicate ───────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    """Stable MD5 hash of the first 500 characters of text."""
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()


@asset(
    group_name="sanitization_observability",
    io_manager_key="hf_parquet_io_manager",
)
def deduplicated_fineweb_edu(
    context: AssetExecutionContext,
    filtered_fineweb_edu: Dataset,
) -> Dataset:
    """Remove near-duplicate documents using a prefix hash.

    Hashes the first 500 characters of each document. Documents
    sharing a hash are considered duplicates; only the first
    occurrence is retained.
    """
    before = len(filtered_fineweb_edu)
    seen: set[str] = set()

    def is_unique(example: dict) -> bool:
        h = _text_hash(example["text"])
        if h in seen:
            return False
        seen.add(h)
        return True

    deduped = filtered_fineweb_edu.filter(is_unique)
    after = len(deduped)

    context.log.info(
        "Deduplication: %s rows → %s rows (%s duplicates removed)",
        before,
        after,
        before - after,
    )

    return deduped


# ── Step 4: Quality report asset ──────────────────────────────────────────────

@asset(
    group_name="sanitization_observability",
)
def cleaning_quality_report(
    context: AssetExecutionContext,
    raw_fineweb_edu: Dataset,
    deduplicated_fineweb_edu: Dataset,
) -> MaterializeResult:
    """Emit a structured quality report comparing raw vs. cleaned dataset.

    Logged as structured metadata visible in the Dagster UI asset catalog.
    """
    raw_rows = len(raw_fineweb_edu)
    clean_rows = len(deduplicated_fineweb_edu)
    retention_pct = round((clean_rows / raw_rows) * 100, 2) if raw_rows > 0 else 0.0

    report = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "dropped_rows": raw_rows - clean_rows,
        "retention_pct": retention_pct,
    }

    context.log.info("Quality report: %s", report)

    return MaterializeResult(
        value=report,
        metadata={
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "dropped_rows": raw_rows - clean_rows,
            "retention_pct": retention_pct,
        },
    )


# ── Asset checks ──────────────────────────────────────────────────────────────

@asset_check(asset=deduplicated_fineweb_edu, description="No null text values after cleaning")
def check_no_null_text(deduplicated_fineweb_edu: Dataset) -> AssetCheckResult:
    """Verify that no null or empty text values remain post-deduplication."""
    null_count = sum(
        1 for ex in deduplicated_fineweb_edu
        if not ex.get("text") or len(ex["text"].strip()) == 0
    )
    return AssetCheckResult(
        passed=null_count == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"null_text_count": null_count},
    )


@asset_check(asset=deduplicated_fineweb_edu, description="Dataset retains at least 80% of raw rows")
def check_retention_rate(
    raw_fineweb_edu: Dataset,
    deduplicated_fineweb_edu: Dataset,
) -> AssetCheckResult:
    """Warn if more than 20% of rows were dropped during cleaning.

    A high drop rate may indicate overly aggressive filtering
    or an unexpected upstream data quality issue.
    """
    raw_rows = len(raw_fineweb_edu)
    clean_rows = len(deduplicated_fineweb_edu)
    retention_pct = (clean_rows / raw_rows * 100) if raw_rows > 0 else 0.0

    return AssetCheckResult(
        passed=retention_pct >= 80.0,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "retention_pct": round(retention_pct, 2),
        },
    )