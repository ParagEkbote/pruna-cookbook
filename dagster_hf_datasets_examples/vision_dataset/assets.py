from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset


@hf_dataset_asset(
    path="google-research-datasets/conceptual_captions",
    split="train",
    group_name="vision_language_dataset",
    io_manager_key="hf_parquet_io_manager",
)
def conceptual_captions(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Load Conceptual Captions."""

    return MaterializeResult(
        value=dataset,
        metadata={
            "rows": len(dataset),
            "columns": dataset.column_names,
            "fingerprint": dataset._fingerprint,
        },
    )


@asset(
    group_name="vision_language_dataset",
    io_manager_key="hf_parquet_io_manager",
)
def validated_pairs(
    context: AssetExecutionContext,
    conceptual_captions: Dataset,
) -> MaterializeResult:
    """Validate image-caption pairs."""

    validated = conceptual_captions.filter(
        lambda ex: (
            ex.get("caption") is not None
            and len(ex["caption"].strip()) > 0
        )
    )

    return MaterializeResult(
        value=validated,
        metadata={
            "validated_rows": len(validated),
        },
    )


@asset(
    group_name="vision_language_dataset",
    io_manager_key="hf_parquet_io_manager",
)
def cc_train(
    validated_pairs: Dataset,
) -> MaterializeResult:
    split = validated_pairs.train_test_split(
        test_size=0.1,
        seed=42,
    )

    return MaterializeResult(
        value=split["train"],
        metadata={
            "rows": len(split["train"]),
            "split": "train",
        },
    )


@asset(
    group_name="vision_language_dataset",
    io_manager_key="hf_parquet_io_manager",
)
def cc_validation(
    validated_pairs: Dataset,
) -> MaterializeResult:
    split = validated_pairs.train_test_split(
        test_size=0.1,
        seed=42,
    )

    return MaterializeResult(
        value=split["test"],
        metadata={
            "rows": len(split["test"]),
            "split": "validation",
        },
    )


@asset(group_name="vision_language_dataset")
def dataset_card(
    cc_train: Dataset,
    cc_validation: Dataset,
) -> MaterializeResult:
    card = f"""
# Vision Language Dataset

Train Rows: {len(cc_train)}
Validation Rows: {len(cc_validation)}

Generated via dagster_hf_datasets.
"""

    return MaterializeResult(
        value=card,
        metadata={
            "train_rows": len(cc_train),
            "validation_rows": len(cc_validation),
        },
    )
