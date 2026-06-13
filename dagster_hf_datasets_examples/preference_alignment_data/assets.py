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
    Prepare an RLHF/DPO-ready preference dataset from hh-rlhf.

    Demonstrates:
    - Preference pair validation
    - Chosen/rejected response checks
    - Alignment dataset preparation

    Dataset: Anthropic/hh-rlhf (foundational RLHF data from Anthropic)
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


@hf_dataset_asset(
    path="allenai/ultrafeedback_binarized_cleaned",
    split="train_prefs",
    group_name="preference_alignment",
    io_manager_key="hf_parquet_io_manager",
)
def ultrafeedback_preference_dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """
    Prepare UltraFeedback preference pairs for DPO/ORPO training.

    Demonstrates:
    - Model-agnostic preference pair preparation
    - Alternative to hh-rlhf with higher quality
    - DPO-format validation

    Dataset: allenai/ultrafeedback_binarized_cleaned
    - 64K+ high-quality preference pairs
    - Annotated by multiple models for diversity
    - Better generalization than model-specific RLHF data
    """

    initial_rows = len(dataset)

    def is_valid(example):
        """Validate UltraFeedback structure (format may differ from hh-rlhf)."""
        # UltraFeedback has 'instruction', 'chosen', 'rejected' fields
        instruction = example.get("instruction")
        chosen = example.get("chosen")
        rejected = example.get("rejected")

        if instruction is None or not instruction.strip():
            return False

        if chosen is None or not chosen.strip():
            return False

        if rejected is None or not rejected.strip():
            return False

        if chosen == rejected:
            return False

        return True

    validated_dataset = dataset.filter(is_valid)

    removed_rows = initial_rows - len(validated_dataset)

    context.log.info(
        "Validated UltraFeedback: kept %s / %s pairs (removed %s malformed)",
        len(validated_dataset),
        initial_rows,
        removed_rows,
    )

    return MaterializeResult(
        value=validated_dataset,
        metadata={
            "original_rows": initial_rows,
            "validated_rows": len(validated_dataset),
            "removed_rows": removed_rows,
            "dataset": "allenai/ultrafeedback_binarized_cleaned",
            "split": "train",
            "quality_note": "High-quality, model-agnostic preference pairs",
            "fingerprint": validated_dataset._fingerprint,
        },
    )
