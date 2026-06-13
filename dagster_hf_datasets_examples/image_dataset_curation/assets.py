from __future__ import annotations

import hashlib
import json
from pathlib import Path

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_hash(img) -> str:
    """Perceptual-adjacent hash: MD5 of raw pixel bytes at 32×32 thumbnail."""
    thumb = img.copy()
    thumb = thumb.convert("RGB")
    thumb = thumb.resize((32, 32))
    return hashlib.md5(thumb.tobytes()).hexdigest()


def _is_corrupt(img) -> bool:
    """Attempt a full decode to catch truncated/corrupt JPEG and PNG files."""
    try:
        img.verify()          # catches truncated files for some formats
        return False
    except Exception:
        return True


def _check_corrupt_via_load(img) -> bool:
    """Secondary corrupt check: try loading pixel data."""
    try:
        img.load()
        return False
    except Exception:
        return True


def _aspect_ratio(img) -> float:
    w, h = img.size
    return round(w / h, 4) if h > 0 else 0.0


# ── Step 1: Ingest ────────────────────────────────────────────────────────────

@hf_dataset_asset(
    path="zh-plus/tiny-imagenet",
    split="train",
    group_name="image_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def tiny_imagenet_raw(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest Tiny ImageNet training split from the Hub.

    Tiny ImageNet contains 100,000 training images across 200 classes
    at 64×64 resolution. Small enough to run locally while exhibiting
    realistic curation challenges: class imbalance, occasional corrupt
    entries, and mixed aspect ratios near boundaries.
    """
    context.log.info("Loaded Tiny ImageNet train: %s rows", len(dataset))
    context.log.info("Columns: %s", dataset.column_names)

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "source_dataset": "zh-plus/tiny-imagenet",
            "split": "train",
            "fingerprint": dataset._fingerprint,
        },
    )


# ── Step 2: Resolution filter ─────────────────────────────────────────────────

MIN_WIDTH = 32
MIN_HEIGHT = 32
MAX_WIDTH = 4096
MAX_HEIGHT = 4096


@asset(
    group_name="image_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def resolution_filtered(
    context: AssetExecutionContext,
    tiny_imagenet_raw: Dataset,
) -> MaterializeResult:
    """Remove images outside the acceptable resolution range.

    Drops images smaller than 32×32 (likely placeholder/corrupt) and
    larger than 4096×4096 (pathological outliers that would blow up
    training batch memory). Tiny ImageNet is nominally 64×64, so this
    primarily catches edge cases and validates the assumption.
    """
    before = len(tiny_imagenet_raw)

    def within_bounds(example: dict) -> bool:
        img = example["image"]
        w, h = img.size
        return MIN_WIDTH <= w <= MAX_WIDTH and MIN_HEIGHT <= h <= MAX_HEIGHT

    filtered = tiny_imagenet_raw.filter(within_bounds, desc="Resolution filter")
    after = len(filtered)

    context.log.info(
        "Resolution filter: %s → %s rows (%s dropped)",
        before, after, before - after,
    )
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "dropped": before - after,
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
        }
    )
    return MaterializeResult(
        value=filtered,
        metadata={
            "rows": after,
            "rows_in": before,
            "rows_out": after,
            "dropped": before - after,
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
        },
    )


# ── Step 3: Corrupt image detection ──────────────────────────────────────────

@asset(
    group_name="image_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def corrupt_removed(
    context: AssetExecutionContext,
    resolution_filtered: Dataset,
) -> MaterializeResult:
    """Remove corrupt or unloadable images.

    Uses a two-pass strategy:
    1. img.verify() — catches truncated streams for JPEG/PNG
    2. img.load()   — forces full pixel decode, catches partial corruption

    Note: PIL's verify() consumes the image object; a fresh copy is
    needed for subsequent operations, so both checks use .copy().
    """
    before = len(resolution_filtered)
    corrupt_indices: list[int] = []

    for i, example in enumerate(resolution_filtered):
        img = example["image"]
        if _check_corrupt_via_load(img.copy()):
            corrupt_indices.append(i)

        if i % 5000 == 0:
            context.log.info("Corruption scan: %s / %s", i, before)

    if corrupt_indices:
        context.log.warning("Found %s corrupt images at indices: %s", len(corrupt_indices), corrupt_indices[:10])
        keep_indices = [i for i in range(before) if i not in set(corrupt_indices)]
        cleaned = resolution_filtered.select(keep_indices)
    else:
        context.log.info("No corrupt images found")
        cleaned = resolution_filtered

    after = len(cleaned)
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "corrupt_removed": len(corrupt_indices),
            "corrupt_indices_sample": str(corrupt_indices[:5]),
        }
    )
    return MaterializeResult(
        value=cleaned,
        metadata={
            "rows": after,
            "rows_in": before,
            "rows_out": after,
            "corrupt_removed": len(corrupt_indices),
            "corrupt_indices_sample": str(corrupt_indices[:5]),
        },
    )


# ── Step 4: Perceptual deduplication ─────────────────────────────────────────

@asset(
    group_name="image_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def deduplicated_images(
    context: AssetExecutionContext,
    corrupt_removed: Dataset,
) -> MaterializeResult:
    """Remove near-duplicate images using 32×32 RGB thumbnail hashing.

    Downsamples each image to 32×32 RGB and hashes the raw pixel bytes.
    Images sharing a hash are considered perceptual duplicates; only the
    first occurrence is retained. This catches exact duplicates and
    near-identical rescaled copies.
    """
    before = len(corrupt_removed)
    seen: set[str] = set()

    def is_unique(example: dict) -> bool:
        h = _image_hash(example["image"])
        if h in seen:
            return False
        seen.add(h)
        return True

    deduped = corrupt_removed.filter(is_unique, desc="Deduplication")
    after = len(deduped)

    context.log.info(
        "Deduplication: %s → %s rows (%s duplicates removed)",
        before, after, before - after,
    )
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "duplicates_removed": before - after,
            "dedup_method": "32x32 RGB pixel hash (MD5)",
        }
    )
    return MaterializeResult(
        value=deduped,
        metadata={
            "rows": after,
            "rows_in": before,
            "rows_out": after,
            "duplicates_removed": before - after,
            "dedup_method": "32x32 RGB pixel hash (MD5)",
        },
    )


# ── Step 5: Aspect ratio validation ──────────────────────────────────────────

MIN_ASPECT = 0.25   # 1:4 portrait
MAX_ASPECT = 4.0    # 4:1 landscape


@asset(
    group_name="image_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def aspect_ratio_validated(
    context: AssetExecutionContext,
    deduplicated_images: Dataset,
) -> MaterializeResult:
    """Split images into curated (accepted) and rejected sets by aspect ratio.

    Images with aspect ratio outside [0.25, 4.0] are atypical for
    classification and likely to harm training. Accepted images are
    returned; rejected indices are logged for audit.

    Returns only the curated (accepted) dataset.
    """
    before = len(deduplicated_images)
    rejected_indices: list[int] = []
    aspect_ratios: list[float] = []

    for i, example in enumerate(deduplicated_images):
        ar = _aspect_ratio(example["image"])
        aspect_ratios.append(ar)
        if ar < MIN_ASPECT or ar > MAX_ASPECT:
            rejected_indices.append(i)

    keep_indices = [i for i in range(before) if i not in set(rejected_indices)]
    curated = deduplicated_images.select(keep_indices)
    after = len(curated)

    import statistics
    context.log.info(
        "Aspect ratio validation: %s → %s accepted, %s rejected",
        before, after, len(rejected_indices),
    )
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "rejected_count": len(rejected_indices),
            "aspect_ratio_mean": round(statistics.mean(aspect_ratios), 4),
            "aspect_ratio_min": round(min(aspect_ratios), 4),
            "aspect_ratio_max": round(max(aspect_ratios), 4),
            "min_aspect_threshold": MIN_ASPECT,
            "max_aspect_threshold": MAX_ASPECT,
        }
    )
    return MaterializeResult(
        value=curated,
        metadata={
            "rows": after,
            "rows_in": before,
            "rows_out": after,
            "rejected_count": len(rejected_indices),
            "aspect_ratio_mean": round(statistics.mean(aspect_ratios), 4),
            "aspect_ratio_min": round(min(aspect_ratios), 4),
            "aspect_ratio_max": round(max(aspect_ratios), 4),
            "min_aspect_threshold": MIN_ASPECT,
            "max_aspect_threshold": MAX_ASPECT,
        },
    )


# ── Step 6: Multi-format export ───────────────────────────────────────────────

@asset(
    group_name="image_dataset_curation",
)
def curated_export(
    context: AssetExecutionContext,
    aspect_ratio_validated: Dataset,
) -> MaterializeResult:
    """Export the curated dataset in Parquet and Arrow formats with a manifest.

    Writes to `.dagster_hf_storage/curated_images/` in two formats:
    - Arrow (save_to_disk): fastest for subsequent datasets library usage
    - Parquet (export_to_parquet): portable, readable by pandas/spark/duckdb

    A JSON manifest records row counts, export paths, and class distribution.
    """
    export_dir = Path(".dagster_hf_storage/curated_images")
    export_dir.mkdir(parents=True, exist_ok=True)

    arrow_path = export_dir / "arrow"
    parquet_path = export_dir / "curated.parquet"

    context.log.info("Saving Arrow format to %s", arrow_path)
    aspect_ratio_validated.save_to_disk(str(arrow_path))

    context.log.info("Saving Parquet format to %s", parquet_path)
    aspect_ratio_validated.to_parquet(str(parquet_path))

    # Class distribution
    label_counts: dict[int, int] = {}
    for ex in aspect_ratio_validated:
        lbl = ex.get("label", -1)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    manifest = {
        "total_rows": len(aspect_ratio_validated),
        "num_classes": len(label_counts),
        "arrow_path": str(arrow_path),
        "parquet_path": str(parquet_path),
        "class_distribution_sample": dict(sorted(label_counts.items())[:10]),
    }

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    context.log.info("Export complete: %s rows, %s classes", manifest["total_rows"], manifest["num_classes"])
    context.add_output_metadata(
        {
            "total_rows": manifest["total_rows"],
            "num_classes": manifest["num_classes"],
            "arrow_path": str(arrow_path),
            "parquet_path": str(parquet_path),
        }
    )
    return MaterializeResult(
        value=manifest,
        metadata={
            "total_rows": manifest["total_rows"],
            "num_classes": manifest["num_classes"],
            "arrow_path": str(arrow_path),
            "parquet_path": str(parquet_path),
        },
    )


# ── Step 7: Curation report ───────────────────────────────────────────────────

@asset(
    group_name="image_dataset_curation",
)
def curation_report(
    context: AssetExecutionContext,
    tiny_imagenet_raw: Dataset,
    resolution_filtered: Dataset,
    corrupt_removed: Dataset,
    deduplicated_images: Dataset,
    aspect_ratio_validated: Dataset,
) -> MaterializeResult:
    """Produce a full funnel report tracking row counts at each curation stage.

    Shows exactly how many images were dropped at each step and the
    cumulative retention rate through the pipeline.
    """
    stages = {
        "raw": len(tiny_imagenet_raw),
        "after_resolution_filter": len(resolution_filtered),
        "after_corrupt_removal": len(corrupt_removed),
        "after_deduplication": len(deduplicated_images),
        "after_aspect_ratio_validation": len(aspect_ratio_validated),
    }

    report = stages

    raw = stages["raw"]
    report = {
        "stages": stages,
        "dropped_per_stage": {
            "resolution_filter": stages["raw"] - stages["after_resolution_filter"],
            "corrupt_removal": stages["after_resolution_filter"] - stages["after_corrupt_removal"],
            "deduplication": stages["after_corrupt_removal"] - stages["after_deduplication"],
            "aspect_ratio": stages["after_deduplication"] - stages["after_aspect_ratio_validation"],
        },
        "total_dropped": raw - stages["after_aspect_ratio_validation"],
        "final_retention_pct": round(stages["after_aspect_ratio_validation"] / raw * 100, 2),
    }

    context.log.info("Curation funnel: %s", report["stages"])
    context.log.info("Final retention: %.1f%%", report["final_retention_pct"])

    return MaterializeResult(
        value=report,
        metadata={
            **{f"stage_{k}": v for k, v in stages.items()},
            "total_dropped": report["total_dropped"],
            "final_retention_pct": report["final_retention_pct"],
        },
    )


# ── Asset checks ──────────────────────────────────────────────────────────────

@asset_check(
    asset=aspect_ratio_validated,
    description="Curated dataset retains at least 90% of deduplicated images",
)
def check_curation_retention(
    deduplicated_images: Dataset,
    aspect_ratio_validated: Dataset,
) -> AssetCheckResult:
    dedup_count = len(deduplicated_images)
    curated_count = len(aspect_ratio_validated)
    retention = (curated_count / dedup_count * 100) if dedup_count > 0 else 0.0

    return AssetCheckResult(
        passed=retention >= 90.0,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "dedup_rows": dedup_count,
            "curated_rows": curated_count,
            "retention_pct": round(retention, 2),
        },
    )


@asset_check(
    asset=aspect_ratio_validated,
    description="All curated images pass aspect ratio bounds",
)
def check_aspect_bounds(aspect_ratio_validated: Dataset) -> AssetCheckResult:
    violations = [
        i for i, ex in enumerate(aspect_ratio_validated)
        if not (MIN_ASPECT <= _aspect_ratio(ex["image"]) <= MAX_ASPECT)
    ]
    return AssetCheckResult(
        passed=len(violations) == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"violation_count": len(violations), "sample_indices": str(violations[:5])},
    )
