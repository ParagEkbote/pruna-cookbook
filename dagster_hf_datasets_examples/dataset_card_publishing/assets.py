from __future__ import annotations

from datetime import datetime, timezone

from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from dagster_hf_datasets._export._publisher import HFDatasetPublisher
from datasets import Dataset


# ── Step 1: Ingest ────────────────────────────────────────────────────────────

@hf_dataset_asset(
    path="rajpurkar/squad",
    split="train",
    group_name="dataset_card_publishing",
    io_manager_key="hf_parquet_io_manager",
)
def squad_train(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest the SQuAD training split from the Hugging Face Hub.

    SQuAD ships a rich existing dataset card — useful as a reference
    for what a well-documented card looks like, and as a base to
    extend with pipeline-derived lineage metadata.
    """
    context.log.info("Loaded SQuAD train: %s rows", len(dataset))

    avg_context_len = sum(len(ex["context"].split()) for ex in dataset) / len(dataset)
    avg_answer_len = sum(len(ex["answers"]["text"][0].split()) for ex in dataset if ex["answers"]["text"]) / len(dataset)

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "avg_context_words": round(avg_context_len, 1),
            "avg_answer_words": round(avg_answer_len, 1),
            "source_dataset": "rajpurkar/squad",
            "split": "train",
            "fingerprint": dataset._fingerprint,
        },
    )


# ── Step 2: Process — add answer length feature ───────────────────────────────

@asset(
    group_name="dataset_card_publishing",
    io_manager_key="hf_parquet_io_manager",
)
def squad_enriched(
    context: AssetExecutionContext,
    squad_train: Dataset,
) -> Dataset:
    """Enrich SQuAD with a derived `answer_length` column.

    Adds the token count of the first answer span to each row.
    This gives the dataset card publisher a concrete processing
    step to document in the Hub README.
    """
    enriched = squad_train.map(
        lambda ex: {
            "answer_length": len(ex["answers"]["text"][0].split())
            if ex["answers"]["text"]
            else 0
        },
        desc="Computing answer lengths",
    )

    max_len = max(ex["answer_length"] for ex in enriched)
    min_len = min(ex["answer_length"] for ex in enriched)
    avg_len = sum(ex["answer_length"] for ex in enriched) / len(enriched)

    context.log.info("Answer length stats — min: %s, max: %s, avg: %.1f", min_len, max_len, avg_len)
    context.add_output_metadata(
        {
            "rows": len(enriched),
            "answer_length_min": min_len,
            "answer_length_max": max_len,
            "answer_length_avg": round(avg_len, 2),
        }
    )

    return enriched


# ── Step 3: Generate and publish dataset card ─────────────────────────────────

@asset(
    group_name="dataset_card_publishing",
)
def squad_dataset_card(
    context: AssetExecutionContext,
    squad_train: Dataset,
    squad_enriched: Dataset,
) -> dict:
    """Generate a Hub dataset card from pipeline metadata and publish it.

    HFDatasetPublisher assembles a README.md from processing step
    descriptions, run metadata, and asset lineage, then calls
    push_to_hub() to publish the card to your Hub namespace.

    The returned dict contains the Hub URL and card content for
    downstream use or logging.

    Set HF_TOKEN and HF_REPO_ID env vars before running.
    """
    import os

    repo_id = os.environ.get("HF_REPO_ID", "your-username/squad-enriched")

    processing_steps = [
        {
            "step": "ingestion",
            "description": "Loaded rajpurkar/squad train split via dagster-hf-datasets",
            "rows_in": len(squad_train),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "step": "enrichment",
            "description": "Added answer_length column (token count of first answer span)",
            "rows_out": len(squad_enriched),
            "new_columns": ["answer_length"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]

    card_metadata = {
        "language": ["en"],
        "task_categories": ["question-answering"],
        "task_ids": ["extractive-qa"],
        "source_datasets": ["rajpurkar/squad"],
        "pipeline": "dagster-hf-datasets",
        "dagster_run_id": context.run_id,
    }

    publisher = HFDatasetPublisher(
        repo_id=repo_id,
        source_dataset="rajpurkar/squad",
        source_revision=squad_train._fingerprint,
        processing_steps=processing_steps,
        metadata=card_metadata,
    )

    # NOTE: Uncomment to publish. Requires HF_TOKEN with write access to repo_id.
    # hub_url = publisher.publish(dataset=squad_enriched)
    # context.log.info("Published to Hub: %s", hub_url)

    # Dry-run: generate the card content without pushing
    card_content = publisher.generate_card()
    context.log.info("Dataset card generated (%s chars)", len(card_content))

    context.add_output_metadata(
        {
            "repo_id": repo_id,
            "card_length_chars": len(card_content),
            "processing_steps": len(processing_steps),
            "dagster_run_id": context.run_id,
        }
    )

    return {
        "repo_id": repo_id,
        "card_content": card_content,
        "processing_steps": processing_steps,
    }