from __future__ import annotations

import json
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
) -> MaterializeResult:
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

    return MaterializeResult(
        value=stats_dataset,
        metadata={
            "rows": len(stats_dataset),
            "image_count": len(records),
            "width_min": min(widths),
            "width_max": max(widths),
            "width_mean": round(statistics.mean(widths), 1),
            "height_min": min(heights),
            "height_max": max(heights),
            "height_mean": round(statistics.mean(heights), 1),
            "aspect_ratio_mean": round(statistics.mean(aspects), 4),
            "color_modes": str(dict(mode_counts)),
        },
    )


# ── Step 3: Caption statistics ────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def caption_stats(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
) -> MaterializeResult:
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

    return MaterializeResult(
        value=stats_dataset,
        metadata={
            "rows": len(stats_dataset),
            "examples_analyzed": len(records),
            "vocabulary_size": vocab_size,
            "avg_caption_length_mean": round(statistics.mean(all_lengths), 2),
            "avg_caption_length_min": round(min(all_lengths), 2),
            "avg_caption_length_max": round(max(all_lengths), 2),
            "total_captions": sum(r["num_captions"] for r in records),
        },
    )


# ── Step 4: Thumbnail gallery ─────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
)
def sample_gallery(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
) -> MaterializeResult:
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

    return MaterializeResult(
        value={"thumbnails": manifest, "gallery_dir": str(gallery_dir)},
        metadata={
            "thumbnail_count": len(manifest),
            "gallery_dir": str(gallery_dir),
            "manifest_path": str(manifest_path),
        },
    )


# ── Step 5: Health report ─────────────────────────────────────────────────────

@asset(
    group_name="multimodal_profiling",
)
def dataset_health_report(
    context: AssetExecutionContext,
    flickr30k_raw: Dataset,
    image_stats: Dataset,
    caption_stats: Dataset,
) -> MaterializeResult:
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

    return MaterializeResult(
        value=report,
        metadata=report,
    )


# ── Step 6: LLaVA-150K (Modern Instruction-Tuned Vision-Language) ────────────

@hf_dataset_asset(
    path="liuhaotian/llava-instruct-150k",
    split="train",
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def llava_instruct_raw(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest LLaVA-150K instruction-tuning dataset.

    LLaVA-150K contains 150K image-instruction-response pairs,
    designed for instruction-tuning vision-language models.
    Unlike raw image-caption pairs (Flickr30K), this data has
    explicit instruction-following structure (Q&A format).

    Modern alternative to generic image-caption datasets for VLM training.
    """
    context.log.info("Loaded LLaVA-150K: %s rows", len(dataset))
    context.log.info("Columns: %s", dataset.column_names)

    return MaterializeResult(
        value=dataset.select(range(min(5000, len(dataset)))),
        metadata={
            "rows": min(5000, len(dataset)),
            "columns": dataset.column_names,
            "source_dataset": "liuhaotian/llava-instruct-150k",
            "split": "train",
            "description": "Instruction-tuning data for vision-language models",
        },
    )


@asset(
    group_name="multimodal_profiling",
    io_manager_key="hf_parquet_io_manager",
)
def llava_instruction_stats(
    context: AssetExecutionContext,
    llava_instruct_raw: Dataset,
) -> MaterializeResult:
    """Extract instruction and response statistics from LLaVA data.

    Unlike Flickr30K (multiple captions per image), LLaVA has
    instruction-response pairs. This asset computes:
    - Instruction complexity (token count, question type)
    - Response length distribution
    - Instruction-response alignment
    """
    records = []

    for i, example in enumerate(llava_instruct_raw):
        # LLaVA structure: 'image', 'conversations' (list of turns)
        conversations = example.get("conversations", [])

        # Typically alternates user (instruction) and assistant (response)
        user_turn = None
        assistant_turn = None

        for turn in conversations:
            role = turn.get("from")
            text = turn.get("value", "").strip()

            if role == "human":
                user_turn = text
            elif role == "gpt":
                assistant_turn = text

        if user_turn and assistant_turn:
            records.append(
                {
                    "idx": i,
                    "instruction_tokens": len(user_turn.split()),
                    "response_tokens": len(assistant_turn.split()),
                    "instruction_length": len(user_turn),
                    "response_length": len(assistant_turn),
                    "is_question": user_turn.rstrip().endswith("?"),
                }
            )

        if i % 1000 == 0:
            context.log.info("Processed %s / %s examples", i, len(llava_instruct_raw))

    instr_tokens = [r["instruction_tokens"] for r in records if records]
    resp_tokens = [r["response_tokens"] for r in records if records]

    context.log.info(
        "Instruction length — min: %s, max: %s, mean: %.1f",
        min(instr_tokens) if instr_tokens else 0,
        max(instr_tokens) if instr_tokens else 0,
        statistics.mean(instr_tokens) if instr_tokens else 0,
    )
    context.log.info(
        "Response length — min: %s, max: %s, mean: %.1f",
        min(resp_tokens) if resp_tokens else 0,
        max(resp_tokens) if resp_tokens else 0,
        statistics.mean(resp_tokens) if resp_tokens else 0,
    )

    stats_dataset = Dataset.from_list(records)

    context.add_output_metadata(
        {
            "example_count": len(records),
            "instruction_tokens_mean": round(statistics.mean(instr_tokens), 1) if instr_tokens else 0,
            "response_tokens_mean": round(statistics.mean(resp_tokens), 1) if resp_tokens else 0,
            "question_fraction": round(sum(1 for r in records if r["is_question"]) / len(records), 2) if records else 0,
        }
    )

    return MaterializeResult(
        value=stats_dataset,
        metadata={
            "rows": len(stats_dataset),
            "example_count": len(records),
            "instruction_tokens_mean": round(statistics.mean(instr_tokens), 1)
            if instr_tokens else 0,
            "response_tokens_mean": round(statistics.mean(resp_tokens), 1)
            if resp_tokens else 0,
            "question_fraction": round(
                sum(1 for r in records if r["is_question"]) / len(records),
                2,
            ) if records else 0,
        },
    )


@asset(group_name="multimodal_profiling")
def llava_quality_profile(
    context: AssetExecutionContext,
    llava_instruct_raw: Dataset,
    llava_instruction_stats: Dataset,
) -> MaterializeResult:
    """Profile quality metrics specific to instruction-tuned VLM data.

    Checks:
    - Instruction-response pair validity
    - Length balance (not too short, not excessively long)
    - Response quality indicators
    """
    total = len(llava_instruct_raw)
    stats_records = list(llava_instruction_stats)

    # Check response quality
    very_short_responses = sum(1 for r in stats_records if r["response_tokens"] < 5)
    very_long_responses = sum(1 for r in stats_records if r["response_tokens"] > 500)
    balanced_responses = total - very_short_responses - very_long_responses

    # Instruction diversity
    questions = sum(1 for r in stats_records if r["is_question"])
    question_pct = round(questions / len(stats_records) * 100, 1) if stats_records else 0

    profile = {
        "total_examples": total,
        "valid_instruction_response_pairs": len(stats_records),
        "very_short_responses_count": very_short_responses,
        "very_long_responses_count": very_long_responses,
        "balanced_responses": balanced_responses,
        "balanced_response_pct": round(balanced_responses / total * 100, 2),
        "question_percentage": question_pct,
        "instruction_complexity_score": round(
            statistics.mean([r["instruction_tokens"] for r in stats_records]) / 10, 1
        ) if stats_records else 0,
    }

    context.log.info("LLaVA quality profile: %s", profile)

    return MaterializeResult(
        value=profile,
        metadata=profile,
    )
