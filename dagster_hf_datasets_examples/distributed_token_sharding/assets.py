from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset
from transformers import AutoTokenizer


TOKENIZER = "bert-base-uncased"


@hf_dataset_asset(
    path="HuggingFaceFW/fineweb",
    name="sample-100BT",
    split="train",
    group_name="tokenization_shard_caching",
    io_manager_key="hf_parquet_io_manager",
)
def fineweb_dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
        },
    )


@asset(
    group_name="tokenization_shard_caching",
    io_manager_key="hf_parquet_io_manager",
)
def tokenized_fineweb(
    context: AssetExecutionContext,
    fineweb_dataset: Dataset,
) -> MaterializeResult:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    tokenized = fineweb_dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
        ),
        batched=True,
        batch_size=1000,
    )

    return MaterializeResult(
        value=tokenized,
        metadata={
            "rows": len(tokenized),
            "tokenizer": TOKENIZER,
        },
    )