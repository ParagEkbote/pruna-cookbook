# Multi-Asset Split Routing

Materialize train, validation, and test splits as independently tracked
Dagster assets using `hf_multi_asset`.

## What this example shows

- Using `@hf_multi_asset` to auto-resolve all splits from a `DatasetDict`
- Each split becoming a first-class asset with its own lineage, history, and metadata
- Downstream `@asset` nodes referencing individual splits by name (e.g. `glue_sst2_train`)
- A cross-split report asset consuming all three splits simultaneously
- Label distribution metadata per split visible in the Dagster UI

## Dataset

[`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`sst2` config) — Stanford Sentiment Treebank binary
classification benchmark. Ships with canonical train / validation / test
splits, making split count and row counts predictable.

| Split | Rows | Labels |
|-------|------|--------|
| train | 67,349 | 0 (negative), 1 (positive) |
| validation | 872 | 0, 1 |
| test | 1,821 | -1 (unlabeled) |

> **Note:** The test split labels are `-1` (withheld for the leaderboard).
> The `split_lineage_report` asset surfaces this in its output metadata.

## Key API

```python
@hf_multi_asset(
    path="nyu-mll/glue",
    config="sst2",
    io_manager_key="hf_parquet_io_manager",
)
def glue_sst2(
    context: AssetExecutionContext,
    datasets: dict[str, Dataset],
) -> dict[str, MaterializeResult]:
    ...
```

`hf_multi_asset` calls `datasets.get_dataset_split_names()` at decoration
time and generates one `AssetOut` per split. The function receives
`datasets: dict[str, Dataset]` — a mapping of split name to loaded dataset.
The return value must be `dict[str, MaterializeResult]` keyed by split name.

### Referencing individual splits downstream

Split assets are named `{asset_name}_{split}`, so downstream assets
declare their dependencies as:

```python
@asset
def my_downstream(glue_sst2_train: Dataset, glue_sst2_validation: Dataset):
    ...
```

## Asset graph

```
              glue_sst2
           /      |      \
    _train  _validation  _test
       |                   |
glue_sst2_train_normalized  \
                             \
                      split_lineage_report (consumes all 3)
```

## Storage layout

```
.dagster_hf_storage/
├── glue_sst2_train/
├── glue_sst2_validation/
├── glue_sst2_test/
└── glue_sst2_train_normalized/
```

## How to run

```bash
pip install dagster dagster-hf-datasets
dagster dev -f definitions.py
```

Materialize `glue_sst2` first (all three splits in one run), then
`glue_sst2_train_normalized` and `split_lineage_report` downstream.
Individual splits can also be re-materialized independently.