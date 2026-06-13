import hashlib
import statistics

from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset

# ── Large-Scale Vision-Language Data Processing ─────────────────────────────


@hf_dataset_asset(
    path="laion/LAION-COCO",
    split="train",
    group_name="vision_language_scale",
    io_manager_key="hf_parquet_io_manager",
)
def raw_laion_coco(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest LAION-COCO: 600M high-quality image-text pairs.

    LAION-COCO is a curated, high-quality subset of LAION designed
    for vision-language model training and instruction-tuning.
    It balances scale (600M pairs) with quality (filtered for coherent
    image-caption pairs).

    Used for training:
    - CLIP alternatives and improvements
    - Vision-language instruction-tuned models (LLaVA, Qwen-VL, etc.)
    - Multimodal understanding models

    Demonstrates:
    - Large-scale multimodal data handling
    - URL-based image references (lazy loading)
    - Metadata-rich caption analysis
    - Sampling strategies for development

    Sample size: 5K pairs for development (adjust as needed)
    """
    sample_size = min(5000, len(dataset))
    sampled = dataset.select(range(sample_size))

    # Analyze caption characteristics in sample
    caption_lengths = []
    for i in range(min(500, len(sampled))):
        caption = sampled[i].get("caption", "")
        caption_lengths.append(len(caption.split()))

    avg_length = statistics.mean(caption_lengths) if caption_lengths else 0
    median_length = statistics.median(caption_lengths) if caption_lengths else 0

    context.log.info(
        "Loaded LAION-COCO sample: %s pairs, caption length: %.1f (avg), %.0f (median)",
        len(sampled),
        avg_length,
        median_length,
    )

    # Sample captions for metadata preview
    sample_captions = [sampled[i].get("caption", "")[:100] for i in range(min(3, len(sampled)))]

    return MaterializeResult(
        value=sampled,
        metadata={
            "rows": len(sampled),
            "total_dataset_size": 600000000,
            "avg_caption_tokens": round(avg_length, 1),
            "sample_captions": sample_captions,
            "source_dataset": "laion/LAION-COCO",
            "modality": "image-text",
        },
    )


@asset(
    group_name="vision_language_scale",
    io_manager_key="hf_parquet_io_manager",
)
def caption_quality_filtered(
    context: AssetExecutionContext,
    raw_laion_coco: Dataset,
) -> MaterializeResult:
    """Filter captions for quality: length, content, and validity.

    Filtering rules:
    - Minimum 3 words (skip single-word or 2-word captions)
    - Maximum 500 words (skip excessively long/off-topic captions)
    - Non-empty after stripping whitespace
    - Exclude boilerplate/placeholder text
    """

    def is_quality_caption(example):
        caption = example.get("caption", "").strip()

        # Length checks
        words = caption.split()
        if len(words) < 3:
            return False
        if len(words) > 500:
            return False

        # Reject known boilerplate
        lower_caption = caption.lower()
        boilerplate = ["website", "banner", "advertisement", "click here", "image not found"]
        if any(phrase in lower_caption for phrase in boilerplate):
            return False

        return True

    filtered = raw_laion_coco.filter(is_quality_caption)

    retention_pct = round((len(filtered) / len(raw_laion_coco)) * 100, 2)

    context.log.info(
        "Quality filtered LAION-COCO: %s → %s pairs (%.1f%% retained)",
        len(raw_laion_coco),
        len(filtered),
        retention_pct,
    )

    context.add_output_metadata(
        {
            "input_rows": len(raw_laion_coco),
            "output_rows": len(filtered),
            "retention_pct": retention_pct,
        }
    )

    return MaterializeResult(value=filtered, metadata={"rows": len(filtered)})


@asset(
    group_name="vision_language_scale",
    io_manager_key="hf_parquet_io_manager",
)
def deduplicated_by_caption_hash(
    context: AssetExecutionContext,
    caption_quality_filtered: Dataset,
) -> MaterializeResult:
    """Remove near-duplicate captions using SHA256 hashing.

    At 600M scale, even 1% duplicates = 6 million redundant pairs.
    Deduplication by caption text hash removes exact duplicates
    (same caption, different image URLs).

    Demonstrates:
    - Deduplication at scale
    - Hash-based tracking (memory efficient)
    - Duplicate statistics for quality reporting
    """
    seen_hashes = set()
    duplicate_count = 0
    deduplicated_indices = []

    for i, example in enumerate(caption_quality_filtered):
        caption = example.get("caption", "").strip()
        caption_hash = hashlib.sha256(caption.encode("utf-8")).hexdigest()

        if caption_hash not in seen_hashes:
            seen_hashes.add(caption_hash)
            deduplicated_indices.append(i)
        else:
            duplicate_count += 1

        if i % 1000 == 0:
            context.log.info(
                "Dedup progress: %s / %s (found %s duplicates)",
                i,
                len(caption_quality_filtered),
                duplicate_count,
            )

    deduped = caption_quality_filtered.select(deduplicated_indices)

    dedup_pct = round((duplicate_count / len(caption_quality_filtered)) * 100, 2)

    context.log.info(
        "Deduplication complete: %s → %s pairs (removed %s duplicates, %.2f%%)",
        len(caption_quality_filtered),
        len(deduped),
        duplicate_count,
        dedup_pct,
    )

    context.add_output_metadata(
        {
            "input_rows": len(caption_quality_filtered),
            "output_rows": len(deduped),
            "duplicates_removed": duplicate_count,
            "duplicate_pct": dedup_pct,
        }
    )

    return MaterializeResult(value=deduped, metadata={"rows": len(deduped)})


@asset(
    group_name="vision_language_scale",
    io_manager_key="hf_parquet_io_manager",
)
def language_identified_captions(
    context: AssetExecutionContext,
    deduplicated_by_caption_hash: Dataset,
) -> MaterializeResult:
    """Identify caption language and filter to English (optional).

    Uses simple heuristics for language detection:
    - Character set analysis
    - Common word matching (can integrate `langdetect` or `textblob` in production)

    This example filters to English; can modify to keep multilingual.
    """

    def detect_language(text):
        """Simple language detection based on character ranges."""
        # English: mostly ASCII + common punctuation
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)

        if ascii_ratio > 0.9:
            return "English"
        elif ascii_ratio > 0.7:
            return "Mixed"
        else:
            return "NonEnglish"

    def is_english(example):
        caption = example.get("caption", "")
        lang = detect_language(caption)
        return lang == "English"

    english_only = deduplicated_by_caption_hash.filter(is_english)

    retention_pct = round((len(english_only) / len(deduplicated_by_caption_hash)) * 100, 2)

    context.log.info(
        "Language filter: %s → %s English captions (%.1f%% retained)",
        len(deduplicated_by_caption_hash),
        len(english_only),
        retention_pct,
    )

    context.add_output_metadata(
        {
            "input_rows": len(deduplicated_by_caption_hash),
            "output_rows": len(english_only),
            "english_retention_pct": retention_pct,
        }
    )

    return MaterializeResult(value=english_only, metadata={"rows": len(english_only)})


@asset(group_name="vision_language_scale")
def dedup_quality_report(
    context: AssetExecutionContext,
    raw_laion_coco: Dataset,
    caption_quality_filtered: Dataset,
    deduplicated_by_caption_hash: Dataset,
    language_identified_captions: Dataset,
) -> dict:
    """Generate comprehensive pipeline efficiency and quality metrics.

    Tracks data through all filtering stages and computes:
    - Stage-by-stage retention %
    - Overall quality score
    - Caption statistics
    """

    # Compute caption length distribution on final dataset (sample)
    sample_size = min(10000, len(language_identified_captions))
    caption_lengths = []

    for i in range(sample_size):
        caption = language_identified_captions[i].get("caption", "")
        caption_lengths.append(len(caption.split()))

    avg_length = statistics.mean(caption_lengths) if caption_lengths else 0
    median_length = statistics.median(caption_lengths) if caption_lengths else 0
    max_length = max(caption_lengths) if caption_lengths else 0
    min_length = min(caption_lengths) if caption_lengths else 0

    total_raw = len(raw_laion_coco)
    total_final = len(language_identified_captions)
    total_retained_pct = round((total_final / total_raw) * 100, 2)

    report = {
        "pipeline_stage": "complete",
        "raw_pairs": total_raw,
        "after_quality_filter": len(caption_quality_filtered),
        "after_deduplication": len(deduplicated_by_caption_hash),
        "after_language_filter": len(language_identified_captions),
        "total_retention_pct": total_retained_pct,
        "quality_filter_retention_pct": round(
            (len(caption_quality_filtered) / len(raw_laion_coco)) * 100, 2
        ),
        "deduplication_retention_pct": round(
            (len(deduplicated_by_caption_hash) / len(caption_quality_filtered)) * 100, 2
        ),
        "language_filter_retention_pct": round(
            (len(language_identified_captions) / len(deduplicated_by_caption_hash)) * 100, 2
        ),
        "caption_length_stats": {
            "mean_tokens": round(avg_length, 1),
            "median_tokens": round(median_length, 1),
            "min_tokens": min_length,
            "max_tokens": max_length,
        },
        "data_quality_score": round(total_retained_pct * 0.8, 1),  # Weighted by retention
    }

    context.log.info("Pipeline complete. Quality report: %s", report)
    context.add_output_metadata(report)

    return report
