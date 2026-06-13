from dagster import AssetExecutionContext, MaterializeResult
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


@hf_dataset_asset(
    path="stanfordnlp/imdb",
    split="train",
    group_name="basic_hub_ingestion",
    io_manager_key="hf_parquet_io_manager",
)
def imdb_train(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Load the IMDb training split from the Hugging Face Hub.

    Materializes the raw dataset as a Dagster asset and attaches
    Hub metadata (row count, columns, fingerprint, revision) for
    lineage tracking in the Dagster UI.
    """
    context.log.info("Loaded IMDb train split: %s rows", len(dataset))
    context.log.info("Columns: %s", dataset.column_names)

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "source_dataset": "stanfordnlp/imdb",
            "split": "train",
            "fingerprint": dataset._fingerprint,
        },
    )


@hf_dataset_asset(
    path="stanfordnlp/imdb",
    split="test",
    group_name="basic_hub_ingestion",
    io_manager_key="hf_parquet_io_manager",
)
def imdb_test(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Load the IMDb test split from the Hugging Face Hub."""
    context.log.info("Loaded IMDb test split: %s rows", len(dataset))

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "source_dataset": "stanfordnlp/imdb",
            "split": "test",
            "fingerprint": dataset._fingerprint,
        },
    )
