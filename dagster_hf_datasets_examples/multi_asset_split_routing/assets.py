from dagster import AssetExecutionContext, AssetOut, MaterializeResult
from dagster_hf_datasets import hf_multi_asset
from datasets import Dataset


# ── Multi-split ingestion ─────────────────────────────────────────────────────
#
# hf_multi_asset resolves available splits at decoration time via
# datasets.get_dataset_split_names() and creates one AssetOut per split.
# Each split is independently tracked, versioned, and materializable.
#
# glue/sst2 ships train / validation / test — all three are wired automatically.

@hf_multi_asset(
    path="nyu-mll/glue",
    config="sst2",
    group_name="multi_asset_splits",
    io_manager_key="hf_parquet_io_manager",
)
def glue_sst2(
    context: AssetExecutionContext,
    datasets: dict[str, Dataset],
) -> dict[str, MaterializeResult]:
    """Materialize the GLUE SST-2 benchmark as independently tracked split assets.

    hf_multi_asset resolves train/validation/test splits from the Hub
    and emits one Dagster asset per split. Each can be individually
    materialized, versioned, and referenced by downstream assets.
    """
    results = {}

    for split_name, dataset in datasets.items():
        context.log.info("Split '%s': %s rows, columns: %s", split_name, len(dataset), dataset.column_names)

        label_counts: dict[int, int] = {}
        for example in dataset:
            label = example.get("label", -1)
            label_counts[label] = label_counts.get(label, 0) + 1

        results[split_name] = MaterializeResult(
            value=dataset,
            metadata={
                "split": split_name,
                "rows": len(dataset),
                "columns": dataset.column_names,
                "label_distribution": str(label_counts),
                "source_dataset": "nyu-mll/glue",
                "config": "sst2",
                "fingerprint": dataset._fingerprint,
            },
        )

    return results


# ── Per-split downstream assets ───────────────────────────────────────────────
#
# Downstream assets reference individual splits by name.
# This demonstrates the asset graph visibility benefit of hf_multi_asset:
# each split has its own lineage, checks, and materialization history.

from dagster import asset
from datasets import ClassLabel


@asset(
    group_name="multi_asset_splits",
    io_manager_key="hf_parquet_io_manager",
)
def glue_sst2_train_normalized(
    context: AssetExecutionContext,
    glue_sst2_train: Dataset,
) -> Dataset:
    """Normalize sentence text in the train split.

    Strips leading/trailing whitespace and lowercases all sentences.
    Demonstrates how individual splits from hf_multi_asset flow
    independently into downstream transformation assets.
    """
    before = len(glue_sst2_train)

    normalized = glue_sst2_train.map(
        lambda ex: {"sentence": ex["sentence"].strip().lower()},
        desc="Normalizing text",
    )

    context.log.info("Normalized %s train rows", before)
    context.add_output_metadata({"rows": before, "transformation": "strip + lowercase"})

    return normalized


@asset(
    group_name="multi_asset_splits",
)
def split_lineage_report(
    context: AssetExecutionContext,
    glue_sst2_train: Dataset,
    glue_sst2_validation: Dataset,
    glue_sst2_test: Dataset,
) -> dict:
    """Emit a cross-split lineage report comparing row counts and label coverage.

    Consumes all three split assets simultaneously, demonstrating that
    hf_multi_asset outputs are independently addressable in the asset graph.
    """
    splits = {
        "train": glue_sst2_train,
        "validation": glue_sst2_validation,
        "test": glue_sst2_test,
    }

    report = {}
    for name, ds in splits.items():
        labels = [ex["label"] for ex in ds]
        unique_labels = sorted(set(labels))
        report[name] = {
            "rows": len(ds),
            "unique_labels": unique_labels,
            "label_counts": {
                str(lbl): labels.count(lbl) for lbl in unique_labels
            },
        }

    total_rows = sum(v["rows"] for v in report.values())
    context.log.info("Total rows across all splits: %s", total_rows)
    context.log.info("Split report: %s", report)

    context.add_output_metadata(
        {
            "train_rows": report["train"]["rows"],
            "validation_rows": report["validation"]["rows"],
            "test_rows": report["test"]["rows"],
            "total_rows": total_rows,
        }
    )

    return report