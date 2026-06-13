from __future__ import annotations

import statistics

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

# ── Config ─────────────────────────────────────────────────────────────────
#
# SAMPLE_SIZE controls CPU runtime. BLIP-base captioning takes roughly
# 0.5-2s/image on CPU, so 40 images runs in ~1-2 minutes. Increase for
# a richer demo; runtime scales roughly linearly.

SAMPLE_SIZE = 40
NSFW_THRESHOLD = 0.5      # DiffusionDB image_nsfw / prompt_nsfw scores, 0-1
ALIGNMENT_THRESHOLD = 0.2  # MiniLM cosine similarity, prompt vs. generated caption


# ── Step 1: Ingest pre-generated synthetic images ────────────────────────────
#
# DiffusionDB already contains Stable-Diffusion-generated images paired
# with the prompts used to create them. This sidesteps running diffusion
# inference entirely — there's no GPU step in this pipeline. The "synthetic"
# framing comes from the images themselves (AI-generated), not from any
# generation step performed here.

@hf_dataset_asset(
    path="poloclub/diffusiondb",
    config="2m_random_1k",
    split="train",
    group_name="synthetic_multimodal_generation",
    io_manager_key="hf_parquet_io_manager",
)
def diffusiondb_sample(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest a small sample of pre-generated Stable Diffusion images + prompts.

    DiffusionDB's `2m_random_1k` config provides 1,000 random
    (image, prompt, generation-params) triples. Only the first
    `SAMPLE_SIZE` rows are kept to bound CPU runtime for downstream
    captioning steps.
    """
    n = min(SAMPLE_SIZE, len(dataset))
    sample = dataset.select(range(n))

    context.log.info("Loaded DiffusionDB sample: %s / %s rows", n, len(dataset))
    context.log.info("Columns: %s", sample.column_names)

    return MaterializeResult(
        value=sample,
        metadata={
            "rows": n,
            "source_rows_available": len(dataset),
            "columns": sample.column_names,
            "source_dataset": "poloclub/diffusiondb",
            "config": "2m_random_1k",
            "fingerprint": sample._fingerprint,
        },
    )


# ── Step 2: NSFW-based quality filtering ──────────────────────────────────────
#
# DiffusionDB ships precomputed image_nsfw / prompt_nsfw scores from the
# original collection pipeline. Filtering on these requires no model —
# it's free quality filtering using existing metadata.

@asset(
    group_name="synthetic_multimodal_generation",
    io_manager_key="hf_parquet_io_manager",
)
def nsfw_filtered(
    context: AssetExecutionContext,
    diffusiondb_sample: Dataset,
) -> Dataset:
    """Drop rows flagged by DiffusionDB's precomputed NSFW scores.

    `image_nsfw` and `prompt_nsfw` are floats in [0, 1] (or -1 if not
    scored). Rows scoring >= NSFW_THRESHOLD on either field are dropped.
    Unscored (-1) rows pass through.
    """
    before = len(diffusiondb_sample)

    def is_safe(example: dict) -> bool:
        img_nsfw = example.get("image_nsfw", -1.0)
        prompt_nsfw = example.get("prompt_nsfw", -1.0)
        if img_nsfw is not None and img_nsfw >= NSFW_THRESHOLD:
            return False
        if prompt_nsfw is not None and prompt_nsfw >= NSFW_THRESHOLD:
            return False
        return True

    filtered = diffusiondb_sample.filter(is_safe, desc="NSFW filter")
    after = len(filtered)

    context.log.info("NSFW filter: %s → %s rows (%s dropped)", before, after, before - after)
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "dropped": before - after,
            "nsfw_threshold": NSFW_THRESHOLD,
        }
    )
    return filtered


# ── Step 3: Synthetic caption generation (BLIP, CPU) ──────────────────────────
#
# This is the "VLM evaluation" step from the original spec, inverted:
# rather than using a VLM to *score* generated images, we use a small
# captioning VLM (BLIP-base, ~990MB) to *generate* a caption for each
# image. BLIP-base runs comfortably on CPU for small batches.

_blip_processor = None
_blip_model = None


def _load_blip():
    """Lazily load BLIP captioning model (downloads ~990MB on first run)."""
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model.eval()
    return _blip_processor, _blip_model


@asset(
    group_name="synthetic_multimodal_generation",
    io_manager_key="hf_parquet_io_manager",
)
def synthetic_captions(
    context: AssetExecutionContext,
    nsfw_filtered: Dataset,
) -> Dataset:
    """Generate a caption for each image using BLIP-base (CPU inference).

    Adds a `generated_caption` column. Each caption is generated
    independently — no batching — to keep memory bounded on CPU.
    """
    import torch

    processor, model = _load_blip()
    captions: list[str] = []

    for i, example in enumerate(nsfw_filtered):
        img = example["image"].convert("RGB")
        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

        if i % 10 == 0:
            context.log.info("Captioned %s / %s images", i, len(nsfw_filtered))

    captioned = nsfw_filtered.add_column("generated_caption", captions)

    avg_len = statistics.mean(len(c.split()) for c in captions)
    context.log.info("Captioning complete. Avg caption length: %.1f words", avg_len)
    context.add_output_metadata(
        {
            "rows": len(captioned),
            "model": "Salesforce/blip-image-captioning-base",
            "avg_caption_length_words": round(avg_len, 2),
        }
    )
    return captioned


# ── Step 4: Caption-prompt alignment scoring (MiniLM, CPU) ────────────────────
#
# Automated evaluation step: scores how semantically aligned the
# BLIP-generated caption is with the original DiffusionDB prompt
# using sentence-embedding cosine similarity. all-MiniLM-L6-v2 is
# ~90MB and runs fast on CPU.

_sentence_model = None


def _load_sentence_model():
    """Lazily load the sentence-embedding model (downloads ~90MB on first run)."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _sentence_model


@asset(
    group_name="synthetic_multimodal_generation",
    io_manager_key="hf_parquet_io_manager",
)
def caption_alignment_scores(
    context: AssetExecutionContext,
    synthetic_captions: Dataset,
) -> Dataset:
    """Score semantic alignment between original prompt and generated caption.

    Adds an `alignment_score` column: cosine similarity between
    sentence embeddings of `prompt` and `generated_caption`, in [-1, 1].
    Low scores indicate the BLIP caption diverged significantly from
    what the prompt asked for — useful for flagging generation failures
    or captioning failures.
    """
    from sentence_transformers import util

    model = _load_sentence_model()
    scores: list[float] = []

    for i, example in enumerate(synthetic_captions):
        emb = model.encode(
            [example["prompt"], example["generated_caption"]],
            convert_to_tensor=True,
        )
        score = float(util.cos_sim(emb[0], emb[1]))
        scores.append(score)

        if i % 10 == 0:
            context.log.info("Scored %s / %s pairs", i, len(synthetic_captions))

    scored = synthetic_captions.add_column("alignment_score", scores)

    context.log.info(
        "Alignment scores — min: %.3f, max: %.3f, mean: %.3f",
        min(scores), max(scores), statistics.mean(scores),
    )
    context.add_output_metadata(
        {
            "rows": len(scored),
            "model": "all-MiniLM-L6-v2",
            "alignment_score_min": round(min(scores), 3),
            "alignment_score_max": round(max(scores), 3),
            "alignment_score_mean": round(statistics.mean(scores), 3),
        }
    )
    return scored


# ── Step 5: Final filtered synthetic dataset ──────────────────────────────────

@asset(
    group_name="synthetic_multimodal_generation",
    io_manager_key="hf_parquet_io_manager",
)
def synthetic_dataset_final(
    context: AssetExecutionContext,
    caption_alignment_scores: Dataset,
) -> Dataset:
    """Filter to rows whose generated caption aligns well with the original prompt.

    Rows with `alignment_score < ALIGNMENT_THRESHOLD` are dropped. The
    output is a synthetic image-caption-prompt triple dataset suitable
    for downstream fine-tuning or evaluation harness use.
    """
    before = len(caption_alignment_scores)

    filtered = caption_alignment_scores.filter(
        lambda ex: ex["alignment_score"] >= ALIGNMENT_THRESHOLD,
        desc="Alignment filter",
    )
    after = len(filtered)

    keep_columns = ["image", "prompt", "generated_caption", "alignment_score", "seed", "cfg", "sampler"]
    available = [c for c in keep_columns if c in filtered.column_names]
    drop_columns = [c for c in filtered.column_names if c not in available]
    final = filtered.remove_columns(drop_columns) if drop_columns else filtered

    context.log.info(
        "Alignment filter: %s → %s rows (%s dropped, threshold=%.2f)",
        before, after, before - after, ALIGNMENT_THRESHOLD,
    )
    context.add_output_metadata(
        {
            "rows_in": before,
            "rows_out": after,
            "dropped": before - after,
            "alignment_threshold": ALIGNMENT_THRESHOLD,
            "columns": final.column_names,
        }
    )
    return final


# ── Step 6: Generation report ─────────────────────────────────────────────────

@asset(
    group_name="synthetic_multimodal_generation",
)
def synthetic_generation_report(
    context: AssetExecutionContext,
    diffusiondb_sample: Dataset,
    nsfw_filtered: Dataset,
    caption_alignment_scores: Dataset,
    synthetic_dataset_final: Dataset,
) -> dict:
    """Funnel report across the synthetic generation + filtering pipeline."""
    stages = {
        "raw_sample": len(diffusiondb_sample),
        "after_nsfw_filter": len(nsfw_filtered),
        "after_captioning": len(caption_alignment_scores),
        "after_alignment_filter": len(synthetic_dataset_final),
    }

    scores = [ex["alignment_score"] for ex in caption_alignment_scores]

    report = {
        "stages": stages,
        "final_retention_pct": round(stages["after_alignment_filter"] / stages["raw_sample"] * 100, 2)
        if stages["raw_sample"] else 0.0,
        "alignment_score_mean": round(statistics.mean(scores), 3) if scores else None,
        "alignment_score_min": round(min(scores), 3) if scores else None,
        "alignment_score_max": round(max(scores), 3) if scores else None,
    }

    context.log.info("Synthetic generation funnel: %s", stages)
    context.add_output_metadata(
        {
            **{f"stage_{k}": v for k, v in stages.items()},
            "final_retention_pct": report["final_retention_pct"],
            "alignment_score_mean": report["alignment_score_mean"],
        }
    )
    return report


# ── Asset checks ──────────────────────────────────────────────────────────────

@asset_check(
    asset=synthetic_dataset_final,
    description="Final dataset has non-empty generated captions for every row",
)
def check_no_empty_captions(synthetic_dataset_final: Dataset) -> AssetCheckResult:
    empty = sum(
        1 for ex in synthetic_dataset_final
        if not ex.get("generated_caption") or len(ex["generated_caption"].strip()) == 0
    )
    return AssetCheckResult(
        passed=empty == 0,
        severity=AssetCheckSeverity.ERROR,
        metadata={"empty_caption_count": empty},
    )


@asset_check(
    asset=synthetic_dataset_final,
    description="Mean alignment score in final dataset is above the filter threshold",
)
def check_mean_alignment(synthetic_dataset_final: Dataset) -> AssetCheckResult:
    if len(synthetic_dataset_final) == 0:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            metadata={"reason": "no rows survived alignment filter"},
        )

    scores = [ex["alignment_score"] for ex in synthetic_dataset_final]
    mean_score = statistics.mean(scores)

    return AssetCheckResult(
        passed=mean_score >= ALIGNMENT_THRESHOLD,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "mean_alignment_score": round(mean_score, 3),
            "threshold": ALIGNMENT_THRESHOLD,
            "rows": len(synthetic_dataset_final),
        },
    )
