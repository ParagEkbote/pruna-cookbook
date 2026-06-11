from dagster import AssetExecutionContext, MaterializeResult
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


@hf_dataset_asset(
    path="Anthropic/hh-rlhf",
    split="train",
    group_name="preference_alignment",
    io_manager_key="hf_parquet_io_manager",
)
def dpo_training_dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """
    Prepare an RLHF/DPO-ready preference dataset.

    Demonstrates:
    - Preference pair validation
    - Chosen/rejected response checks
    - Alignment dataset preparation
    """

    initial_rows = len(dataset)

    def is_valid(example):
        chosen = example.get("chosen")
        rejected = example.get("rejected")

        if chosen is None:
            return False

        if rejected is None:
            return False

        if not chosen.strip():
            return False

        if not rejected.strip():
            return False

        if chosen == rejected:
            return False

        return True

    validated_dataset = dataset.filter(is_valid)

    removed_rows = initial_rows - len(validated_dataset)

    context.log.info(
        "Removed %s malformed preference pairs",
        removed_rows,
    )

    return MaterializeResult(
        value=validated_dataset,
        metadata={
            "original_rows": initial_rows,
            "validated_rows": len(validated_dataset),
            "removed_rows": removed_rows,
            "dataset": "Anthropic/hh-rlhf",
            "split": "train",
            "fingerprint": validated_dataset._fingerprint,
        },
    )