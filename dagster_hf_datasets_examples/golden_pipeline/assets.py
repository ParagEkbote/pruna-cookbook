from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    asset,
)
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset
from transformers import AutoTokenizer

TOKENIZER = "bert-base-uncased"


@hf_dataset_asset(
    path="HuggingFaceFW/fineweb",
    config="sample-10BT",
    split="train",
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_raw(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """
    Ingest FineWeb dataset sample.

    FineWeb is a modern web corpus used by many
    state-of-the-art LLMs.
    """

    sample_size = min(10_000, len(dataset))
    sampled = dataset.select(range(sample_size))

    context.log.info(
        "Loaded FineWeb sample: %s rows",
        sample_size,
    )

    return MaterializeResult(
        value=sampled,
        metadata={
            "rows": sample_size,
            "source_dataset": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "fingerprint": sampled._fingerprint,
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_cleaned(
    context: AssetExecutionContext,
    fineweb_raw: Dataset,
) -> MaterializeResult:
    """
    Remove very short documents.
    """

    original_rows = len(fineweb_raw)

    cleaned = fineweb_raw.filter(
        lambda ex: (
            ex["text"] is not None
            and len(ex["text"].strip()) > 50
        )
    )

    retention_pct = (
        round((len(cleaned) / original_rows) * 100, 2)
        if original_rows > 0
        else 0.0
    )

    context.log.info(
        "Filtered %s → %s rows (%.2f%% retained)",
        original_rows,
        len(cleaned),
        retention_pct,
    )

    return MaterializeResult(
        value=cleaned,
        metadata={
            "original_rows": original_rows,
            "cleaned_rows": len(cleaned),
            "retention_pct": retention_pct,
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_quality_validated(
    context: AssetExecutionContext,
    fineweb_cleaned: Dataset,
) -> MaterializeResult:
    """
    Apply simple quality validation.
    """

    validated = fineweb_cleaned.filter(
        lambda ex: (
            ex["text"] is not None
            and not ex["text"].isspace()
            and len(ex["text"]) < 10_000
        )
    )

    context.log.info(
        "Validated %s rows",
        len(validated),
    )

    return MaterializeResult(
        value=validated,
        metadata={
            "validated_rows": len(validated),
        },
    )


@asset(group_name="golden_master_pipeline")
def quality_report(
    context: AssetExecutionContext,
    fineweb_quality_validated: Dataset,
) -> MaterializeResult:
    """
    Generate dataset quality report.
    """

    report = {
        "dataset": "HuggingFaceFW/fineweb",
        "validated_rows": len(fineweb_quality_validated),
        "fingerprint": fineweb_quality_validated._fingerprint,
    }

    context.log.info("Generated quality report")

    return MaterializeResult(
        value=report,
        metadata=report,
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_train(
    context: AssetExecutionContext,
    fineweb_quality_validated: Dataset,
) -> MaterializeResult:
    """
    Create train split.
    """

    split = fineweb_quality_validated.train_test_split(
        test_size=0.1,
        seed=42,
    )

    train_ds = split["train"]

    return MaterializeResult(
        value=train_ds,
        metadata={
            "rows": len(train_ds),
            "split": "train",
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_test(
    context: AssetExecutionContext,
    fineweb_quality_validated: Dataset,
) -> MaterializeResult:
    """
    Create test split.
    """

    split = fineweb_quality_validated.train_test_split(
        test_size=0.1,
        seed=42,
    )

    test_ds = split["test"]

    return MaterializeResult(
        value=test_ds,
        metadata={
            "rows": len(test_ds),
            "split": "test",
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_train_tokenized(
    context: AssetExecutionContext,
    fineweb_train: Dataset,
) -> MaterializeResult:
    """
    Tokenize train split.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    assert tokenizer is not None

    tokenized = fineweb_train.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
        ),
        batched=True,
        batch_size=1000,
        desc="Tokenizing train split",
    )

    context.log.info(
        "Tokenized train split: %s rows",
        len(tokenized),
    )

    return MaterializeResult(
        value=tokenized,
        metadata={
            "rows": len(tokenized),
            "tokenizer": TOKENIZER,
            "max_length": 512,
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_test_tokenized(
    context: AssetExecutionContext,
    fineweb_test: Dataset,
) -> MaterializeResult:
    """
    Tokenize test split.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    assert tokenizer is not None

    tokenized = fineweb_test.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
        ),
        batched=True,
        batch_size=1000,
        desc="Tokenizing test split",
    )

    context.log.info(
        "Tokenized test split: %s rows",
        len(tokenized),
    )

    return MaterializeResult(
        value=tokenized,
        metadata={
            "rows": len(tokenized),
            "tokenizer": TOKENIZER,
            "max_length": 512,
        },
    )


@asset(group_name="golden_master_pipeline")
def dataset_card(
    fineweb_train_tokenized: Dataset,
    fineweb_test_tokenized: Dataset,
) -> MaterializeResult:
    """
    Generate dataset card.
    """

    card = f"""
# FineWeb Golden Master Dataset

## Dataset Summary

Source: HuggingFaceFW/fineweb
Configuration: sample-10BT

## Splits

Train Rows: {len(fineweb_train_tokenized)}
Test Rows: {len(fineweb_test_tokenized)}

## Tokenization

Tokenizer: {TOKENIZER}

## Pipeline

Raw → Cleaned → Validated → Split → Tokenized

Generated by dagster_hf_datasets.
"""

    return MaterializeResult(
        value=card,
        metadata={
            "train_rows": len(fineweb_train_tokenized),
            "test_rows": len(fineweb_test_tokenized),
        },
    )


@asset(group_name="golden_master_pipeline")
def hub_publication_manifest(
    dataset_card: str,
) -> MaterializeResult:
    """
    Simulated publication manifest.
    """

    manifest = {
        "status": "ready_for_publish",
        "repository": "your-org/golden-master-dataset",
        "source_dataset": "HuggingFaceFW/fineweb",
        "tokenizer": TOKENIZER,
        "pipeline": "golden_master_pipeline",
    }

    return MaterializeResult(
        value=manifest,
        metadata=manifest,
    )
