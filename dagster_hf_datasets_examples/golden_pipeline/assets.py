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
    path="allenai/c4",
    config_name="realnewslike",
    split="train",
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_raw(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    return MaterializeResult(
        value=dataset.select(range(min(10000, len(dataset)))),
        metadata={
            "rows": min(10000, len(dataset)),
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_cleaned(
    c4_raw: Dataset,
) -> MaterializeResult:
    cleaned = c4_raw.filter(
        lambda ex: len(ex["text"].strip()) > 50
    )

    return MaterializeResult(
        value=cleaned,
        metadata={
            "rows": len(cleaned),
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_quality_validated(
    c4_cleaned: Dataset,
) -> MaterializeResult:
    validated = c4_cleaned.filter(
        lambda ex: ex["text"] is not None
    )

    return MaterializeResult(
        value=validated,
        metadata={
            "rows": len(validated),
        },
    )


@asset(group_name="golden_master_pipeline")
def quality_report(
    c4_quality_validated: Dataset,
) -> MaterializeResult:
    return MaterializeResult(
        value={
            "rows": len(c4_quality_validated),
        },
        metadata={
            "rows": len(c4_quality_validated),
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_train(
    c4_quality_validated: Dataset,
) -> MaterializeResult:
    split = c4_quality_validated.train_test_split(
        test_size=0.1,
        seed=42,
    )

    return MaterializeResult(
        value=split["train"],
        metadata={
            "rows": len(split["train"]),
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_test(
    c4_quality_validated: Dataset,
) -> MaterializeResult:
    split = c4_quality_validated.train_test_split(
        test_size=0.1,
        seed=42,
    )

    return MaterializeResult(
        value=split["test"],
        metadata={
            "rows": len(split["test"]),
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_train_tokenized(
    c4_train: Dataset,
) -> MaterializeResult:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    tokenized = c4_train.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
        ),
        batched=True,
    )

    return MaterializeResult(
        value=tokenized,
        metadata={
            "rows": len(tokenized),
            "tokenizer": TOKENIZER,
        },
    )


@asset(
    group_name="golden_master_pipeline",
    io_manager_key="hf_parquet_io_manager",
)
def c4_test_tokenized(
    c4_test: Dataset,
) -> MaterializeResult:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    tokenized = c4_test.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
        ),
        batched=True,
    )

    return MaterializeResult(
        value=tokenized,
        metadata={
            "rows": len(tokenized),
            "tokenizer": TOKENIZER,
        },
    )


@asset(group_name="golden_master_pipeline")
def dataset_card(
    c4_train_tokenized: Dataset,
    c4_test_tokenized: Dataset,
) -> MaterializeResult:
    card = f"""
# Golden Master Dataset

Train Rows: {len(c4_train_tokenized)}
Test Rows: {len(c4_test_tokenized)}

Tokenizer: {TOKENIZER}
"""

    return MaterializeResult(
        value=card,
        metadata={
            "train_rows": len(c4_train_tokenized),
            "test_rows": len(c4_test_tokenized),
        },
    )


@asset(group_name="golden_master_pipeline")
def hub_publication_manifest(
    dataset_card: str,
) -> MaterializeResult:
    manifest = {
        "status": "ready_for_publish",
        "repository": "your-org/golden-master-dataset",
    }

    return MaterializeResult(
        value=manifest,
        metadata=manifest,
    )