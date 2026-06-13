from __future__ import annotations

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    asset,
)
from dagster_hf_datasets import hf_dataset_asset
from dagster_hf_datasets._export._publisher import (
    HFDatasetPublisher,
)
from datasets import Dataset


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
    """
    Load the SQuAD training split.
    """

    context.log.info(
        f"Loaded SQuAD train split with {len(dataset)} rows"
    )

    avg_context_len = (
        sum(
            len(ex["context"].split())
            for ex in dataset
        )
        / len(dataset)
    )

    avg_answer_len = (
        sum(
            len(ex["answers"]["text"][0].split())
            for ex in dataset
            if ex["answers"]["text"]
        )
        / len(dataset)
    )

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "avg_context_words": round(
                avg_context_len,
                1,
            ),
            "avg_answer_words": round(
                avg_answer_len,
                1,
            ),
            "source_dataset": "rajpurkar/squad",
            "split": "train",
            "fingerprint": dataset._fingerprint,
        },
    )


@asset(
    group_name="dataset_card_publishing",
    io_manager_key="hf_parquet_io_manager",
)
def squad_enriched(
    context: AssetExecutionContext,
    squad_train: Dataset,
) -> MaterializeResult:
    """
    Add an answer_length feature.
    """

    enriched = squad_train.map(
        lambda ex: {
            "answer_length": (
                len(
                    ex["answers"]["text"][0].split()
                )
                if ex["answers"]["text"]
                else 0
            )
        },
        desc="Computing answer lengths",
    )

    max_len = max(
        ex["answer_length"]
        for ex in enriched
    )
    min_len = min(
        ex["answer_length"]
        for ex in enriched
    )
    avg_len = (
        sum(
            ex["answer_length"]
            for ex in enriched
        )
        / len(enriched)
    )

    context.log.info(
        f"Answer length stats: "
        f"min={min_len}, "
        f"max={max_len}, "
        f"avg={avg_len:.1f}"
    )

    return MaterializeResult(
        value=enriched,
        metadata={
            "rows": len(enriched),
            "answer_length_min": min_len,
            "answer_length_max": max_len,
            "answer_length_avg": round(
                avg_len,
                2,
            ),
        },
    )


@asset(
    group_name="dataset_card_publishing",
)
def publish_squad_dataset(
    context: AssetExecutionContext,
    squad_train: Dataset,
    squad_enriched: Dataset,
) -> MaterializeResult:
    """
    Publish the enriched dataset and generate a dataset card.
    """

    import os

    repo_id = os.environ.get(
        "HF_REPO_ID",
        "your-username/squad-enriched",
    )

    processing_steps = [
        (
            "Loaded rajpurkar/squad "
            "training split"
        ),
        (
            "Added answer_length column "
            "containing token counts "
            "for the first answer span"
        ),
    ]

    card_metadata = {
        "language": ["en"],
        "task_categories": [
            "question-answering"
        ],
        "task_ids": [
            "extractive-qa"
        ],
        "source_datasets": [
            "rajpurkar/squad"
        ],
        "pipeline": "dagster-hf-datasets",
        "dagster_run_id": context.run_id,
        "derived_features": [
            "answer_length"
        ],
    }

    publisher = HFDatasetPublisher(
        repo_id=repo_id,
        token=os.environ.get("HF_TOKEN"),
    )

    hub_url = publisher.publish(
        dataset=squad_enriched,
        source_dataset="rajpurkar/squad",
        source_revision=(
            squad_train._fingerprint
        ),
        processing_steps=processing_steps,
        metadata=card_metadata,
        description=(
            "SQuAD dataset enriched with "
            "an answer_length feature "
            "generated via a Dagster pipeline."
        ),
    )

    context.log.info(
        f"Published dataset to {hub_url}"
    )

    return MaterializeResult(
        value={
            "repo_id": repo_id,
            "hub_url": hub_url,
        },
        metadata={
            "repo_id": repo_id,
            "hub_url": hub_url,
            "rows": len(squad_enriched),
            "processing_steps": len(
                processing_steps
            ),
            "dagster_run_id": context.run_id,
        },
    )
