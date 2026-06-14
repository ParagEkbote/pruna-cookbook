# Distributed Token Sharding

Tokenize a large web dataset as a Dagster asset and persist the result through
the Hugging Face Parquet IO manager.

## What this example shows

- Loading a large Hub dataset with `@hf_dataset_asset`
- Applying a Hugging Face tokenizer in a downstream `@asset`
- Using batched `Dataset.map()` for tokenization throughput
- Persisting both raw and tokenized datasets with `HFParquetIOManager`
- Recording tokenizer and row-count metadata in the Dagster UI

## Dataset

[`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-100BT` config) - a
large-scale cleaned web corpus used for language-model pretraining experiments.
This example keeps the asset graph intentionally small so the focus stays on
the ingestion -> tokenization handoff.

| Asset | Description |
|-------|-------------|
| `fineweb_dataset` | Loads the FineWeb sample from the Hub |
| `tokenized_fineweb` | Tokenizes the `text` column with `bert-base-uncased` |

## Asset graph

```
fineweb_dataset
      |
      v
tokenized_fineweb
```

## Key API

```python
@asset(
    group_name="tokenization_shard_caching",
    io_manager_key="hf_parquet_io_manager",
)
def tokenized_fineweb(
    context: AssetExecutionContext,
    fineweb_dataset: Dataset,
) -> MaterializeResult:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized = fineweb_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True),
        batched=True,
        batch_size=1000,
    )

    return MaterializeResult(value=tokenized, metadata={"rows": len(tokenized)})
```

`Dataset.map(..., batched=True)` processes multiple records per tokenizer call,
which is the standard pattern for keeping tokenization overhead manageable.

## Metadata visible in the Dagster UI

| Asset | Key | Description |
|-------|-----|-------------|
| `fineweb_dataset` | `rows` | Number of raw rows loaded from the Hub |
| `tokenized_fineweb` | `rows` | Number of rows after tokenization |
| `tokenized_fineweb` | `tokenizer` | Tokenizer used for the transformation |

## Storage layout

```
.dagster_hf_storage/
├── fineweb_dataset/
└── tokenized_fineweb/
```

Both assets are written by `HFParquetIOManager`, so downstream assets can receive
the materialized `Dataset` object directly.

## How to run

```bash
cd dagster_hf_datasets_examples

dagster dev -m distributed_token_sharding.definitions
```

Materialize `fineweb_dataset` first, then `tokenized_fineweb`.

> **Note:** FineWeb configs can be large. For local testing, reduce the dataset
> inside `fineweb_dataset()` before tokenization if you do not want to process
> the full split.
