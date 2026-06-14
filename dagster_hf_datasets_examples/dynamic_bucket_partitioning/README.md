# Dynamic Bucket Partitioning

Register language buckets as Dagster dynamic partitions and materialize a
partition-scoped report for each bucket.

## What this example shows

- Defining a `DynamicPartitionsDefinition`
- Loading a multilingual Hub dataset with `@hf_dataset_asset`
- Modeling language or route buckets as Dagster partitions
- Reading `context.partition_key` inside a partitioned asset
- Keeping raw ingestion separate from per-bucket reporting

## Dataset

[`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) - a multilingual
parallel text dataset built from translated books. The dataset is useful for
examples where language pairs or locales need to become operational routing
keys.

| Asset | Description |
|-------|-------------|
| `opus_books_raw` | Loads the train split from the Hub |
| `partition_report` | Emits metadata for one dynamic partition key |

## Asset graph

```
opus_books_raw

partition_report[language_partitions]
```

`partition_report` is intentionally independent from `opus_books_raw` in this
minimal example. In a production pipeline, the partitioned asset would usually
filter the raw dataset to the selected language bucket.

## Key API

```python
language_partitions = DynamicPartitionsDefinition(
    name="language_partitions"
)

@asset(
    partitions_def=language_partitions,
    group_name="dynamic_bucket_partitioning",
)
def partition_report(context: AssetExecutionContext) -> MaterializeResult:
    partition = context.partition_key

    return MaterializeResult(metadata={"partition": partition})
```

Dynamic partitions are created at runtime, which makes them a good fit for
datasets whose language set, customer set, or source buckets are discovered
after deployment.

## Example partition keys

| Partition key | Meaning |
|---------------|---------|
| `en-fr` | English/French bucket |
| `en-de` | English/German bucket |
| `es-en` | Spanish/English bucket |

Add keys from the Dagster UI before materializing `partition_report`.

## Storage layout

```
.dagster_hf_storage/
└── opus_books_raw/
```

`partition_report` returns metadata only, so it is not persisted by the Hugging
Face IO manager.

## How to run

```bash
cd dagster_hf_datasets_examples

dagster dev -m dynamic_bucket_partitioning.definitions
```

In the Dagster UI, open **Assets**, select `partition_report`, add one or more
dynamic partition keys, then materialize the selected partitions.
