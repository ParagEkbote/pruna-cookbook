from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MaterializeResult,
    asset,
)

from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


language_partitions = DynamicPartitionsDefinition(
    name="language_partitions"
)


@hf_dataset_asset(
    path="Helsinki-NLP/opus_books",
    split="train",
    group_name="dynamic_bucket_partitioning",
    io_manager_key="hf_parquet_io_manager",
)
def opus_books_raw(
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
    partitions_def=language_partitions,
    group_name="dynamic_bucket_partitioning",
)
def partition_report(
    context: AssetExecutionContext,
) -> MaterializeResult:
    partition = context.partition_key

    return MaterializeResult(
        metadata={
            "partition": partition,
        }
    )