# Basic Hub Ingestion

Load a public Hugging Face dataset as a Dagster asset with automatic metadata extraction.

## What this example shows

- Using `@hf_dataset_asset` to declare a Hub-backed Dagster asset
- Returning `MaterializeResult` with dataset metadata (rows, columns, fingerprint)
- Registering `HuggingFaceResource` and `HFParquetIOManager` in `Definitions`
- Materialization of two independent splits (`train`, `test`) as separate assets

## Dataset

[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) — 50 K movie reviews with binary sentiment labels. Small, text-only, no auth required.

| Split | Rows |
|-------|------|
| train | 25,000 |
| test  | 25,000 |

## Key API

```python
@hf_dataset_asset(
    path="stanfordnlp/imdb",
    split="train",
    io_manager_key="hf_parquet_io_manager",
)
def imdb_train(context: AssetExecutionContext, dataset: Dataset) -> MaterializeResult:
    ...
```

> **Note:** The decorated function body is **not** called by Dagster to load the dataset.
> `HuggingFaceResource` performs the load and injects `dataset` as a parameter.
> The function body is where you inspect, log, and return the result.

## Storage layout

After materialization, `HFParquetIOManager` writes:

```
.dagster_hf_storage/
├── imdb_train/          # Arrow format via save_to_disk()
└── imdb_test/
```

## How to run

```bash
pip install dagster dagster-hf-datasets
dagster dev -f definitions.py
```

Then open [http://localhost:3000](http://localhost:3000), navigate to the Asset Catalog,
and materialize `imdb_train` and `imdb_test`.

## Metadata visible in the Dagster UI

| Key | Description |
|-----|-------------|
| `rows` | Row count for the materialized split |
| `columns` | List of column names |
| `source_dataset` | Hub dataset identifier |
| `split` | Which split was loaded |
| `fingerprint` | Reproducibility hash from the datasets library |