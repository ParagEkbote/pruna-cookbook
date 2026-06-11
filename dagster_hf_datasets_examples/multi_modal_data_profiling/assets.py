from __future__ import annotations

import io
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


# ── Step 1: Ingest ────────────────────────────────────────────────────────────

@hf_dataset_asset(
    path="nlphuji/flickr30k",
    split="test",
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def flickr30k_raw(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest Flickr30k from the Hub.

    Flickr30k contains 31,783 images each paired with 5 captions,
    stored as PIL Image objects in the `image` column and lists
    in the `caption` column.
    """
    context.log.info("Loaded Flickr30k: %s rows", len(dataset))
    context.log.info("Columns: %s", dataset.column_names)
    context.log.info("Features: %s", dataset.features)

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "source_dataset": "nlphuji/flickr30k",
            "split": "test",
            "fingerprint": dataset._fingerprint,
        },
    )


# ── Step 2: Image statistics ──────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def image_stats(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
) -> Dataset:
    """Extract per-image resolution and aspect ratio statistics.

    Iterates over PIL Image objects in the dataset, recording
    width, height, aspect ratio, and mode for each image.
    Results are returned as a flat Dataset for downstream use
    and stored via the IO manager.
    """
    records = []

    for i, example in enumerate(flickr30k_raw):
        img = example["image"]  # PIL.Image
        width, height = img.size
        aspect = round(width / height, 4) if height > 0 else 0.0
        records.append(
            {
                "idx": i,
                "width": width,
                "height": height,
                "aspect_ratio": aspect,
                "mode": img.mode,
                "megapixels": round((width * height) / 1_000_000, 4),
            }
        )

        if i % 1000 == 0:
            context.log.info("Processed %s / %s images", i, len(flickr30k_raw))

    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    aspects = [r["aspect_ratio"] for r in records]
    mode_counts = Counter(r["mode"] for r in records)

    context.log.info(
        "Width — min: %s, max: %s, mean: %.1f",
        min(widths), max(widths), statistics.mean(widths),
    )
    context.log.info(
        "Height — min: %s, max: %s, mean: %.1f",
        min(heights), max(heights), statistics.mean(heights),
    )
    context.log.info("Color modes: %s", dict(mode_counts))

    stats_dataset = Dataset.from_list(records)

    context.add_output_metadata(
        {
            "image_count": len(records),
            "width_min": min(widths),
            "width_max": max(widths),
            "width_mean": round(statistics.mean(widths), 1),
            "height_min": min(heights),
            "height_max": max(heights),
            "height_mean": round(statistics.mean(heights), 1),
            "aspect_ratio_mean": round(statistics.mean(aspects), 4),
            "color_modes": str(dict(mode_counts)),
        }
    )

    return stats_dataset


# ── Step 3: Caption statistics ────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def caption_stats(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
) -> Dataset:
    """Compute per-example caption length and vocabulary statistics.

    Each Flickr30k example has a list of 5 captions. This asset
    flattens them and computes token counts, unique word counts,
    and average caption length per example.
    """
    records = []
    all_tokens: list[str] = []

    for i, example in enumerate(flickr30k_raw):
        captions: list[str] = example["caption"]  # list of 5 strings
        token_counts = [len(c.split()) for c in captions]
        avg_tokens = statistics.mean(token_counts)

        all_tokens.extend(tok for c in captions for tok in c.lower().split())

        records.append(
            {
                "idx": i,
                "num_captions": len(captions),
                "avg_caption_length": round(avg_tokens, 2),
                "min_caption_length": min(token_counts),
                "max_caption_length": max(token_counts),
            }
        )

    vocab_size = len(set(all_tokens))
    all_lengths = [r["avg_caption_length"] for r in records]

    context.log.info("Total vocabulary size: %s unique tokens", vocab_size)
    context.log.info(
        "Caption length — min: %.1f, max: %.1f, mean: %.1f",
        min(all_lengths), max(all_lengths), statistics.mean(all_lengths),
    )

    stats_dataset = Dataset.from_list(records)

    context.add_output_metadata(
        {
            "examples_analyzed": len(records),
            "vocabulary_size": vocab_size,
            "avg_caption_length_mean": round(statistics.mean(all_lengths), 2),
            "avg_caption_length_min": round(min(all_lengths), 2),
            "avg_caption_length_max": round(max(all_lengths), 2),
            "total_captions": sum(r["num_captions"] for r in records),
        }
    )

    return stats_dataset


# ── Step 4: Thumbnail gallery ─────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
)
def sample_gallery(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
) -> dict:
    """Save a thumbnail gallery of 16 sample images to disk.

    Writes 128×128 JPEG thumbnails to `.dagster_hf_storage/sample_gallery/`.
    Returns a manifest dict with file paths and the first caption for each.
    """
    gallery_dir = Path(".dagster_hf_storage/sample_gallery")
    gallery_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = list(range(0, min(16, len(flickr30k_raw))))
    manifest = []

    for i in sample_indices:
        example = flickr30k_raw[i]
        img = example["image"].copy()
        img.thumbnail((128, 128))

        out_path = gallery_dir / f"sample_{i:04d}.jpg"
        img.save(out_path, format="JPEG", quality=85)

        manifest.append(
            {
                "idx": i,
                "path": str(out_path),
                "caption": example["caption"][0] if example["caption"] else "",
                "original_size": example["image"].size,
                "thumbnail_size": img.size,
            }
        )

    manifest_path = gallery_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    context.log.info("Saved %s thumbnails to %s", len(manifest), gallery_dir)
    context.add_output_metadata(
        {
            "thumbnail_count": len(manifest),
            "gallery_dir": str(gallery_dir),
            "manifest_path": str(manifest_path),
        }
    )

    return {"thumbnails": manifest, "gallery_dir": str(gallery_dir)}


# ── Step 5: Health report ─────────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
)
def dataset_health_report(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
    image_stats: Dataset,
    caption_stats: Dataset,
) -> dict:
    """Combine image and caption statistics into a single health report.

    Flags potential quality issues:
    - Images with unusual aspect ratios (< 0.2 or > 5.0)
    - Captions shorter than 4 tokens on average
    - Missing or empty captions
    """
    # Aspect ratio outliers
    extreme_aspect = [
        row for row in image_stats
        if row["aspect_ratio"] < 0.2 or row["aspect_ratio"] > 5.0
    ]

    # Short captions
    short_captions = [
        row for row in caption_stats
        if row["avg_caption_length"] < 4.0
    ]

    # Missing captions
    missing_captions = sum(
        1 for ex in flickr30k_raw
        if not ex.get("caption") or len(ex["caption"]) == 0
    )

    total = len(flickr30k_raw)
    report = {
        "total_examples": total,
        "extreme_aspect_ratio_count": len(extreme_aspect),
        "extreme_aspect_ratio_pct": round(len(extreme_aspect) / total * 100, 2),
        "short_caption_count": len(short_captions),
        "short_caption_pct": round(len(short_captions) / total * 100, 2),
        "missing_caption_count": missing_captions,
        "health_score": round(
            100
            - (len(extreme_aspect) / total * 30)
            - (len(short_captions) / total * 40)
            - (missing_captions / total * 30),
            1,
        ),
    }

    context.log.info("Dataset health report: %s", report)
    context.add_output_metadata(report)

    return report