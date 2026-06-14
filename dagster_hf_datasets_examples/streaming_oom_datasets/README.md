# Streaming OOM Datasets

Demonstrate the pattern for working with datasets that are too large to load
fully into memory.

## What this example shows

- Why streaming datasets need different handling from in-memory `Dataset` assets
- Avoiding `len(dataset)` on streamed inputs
- Emitting bounded reports instead of materializing an entire large corpus
- Keeping memory use predictable by processing a fixed number of records
- Documenting the tradeoff between exact row counts and streaming safety

## Dataset pattern

Large text corpora such as Dolma, FineWeb, or Common Crawl derivatives can be
too large for local materialization. In those cases, use Hugging Face streaming
mode and consume only the records needed for the current asset.

| Concern | In-memory dataset | Streaming dataset |
|---------|-------------------|-------------------|
| Row count | `len(dataset)` is available | Often unavailable |
| Access | Random access and full transforms | Iterator-style processing |
| Best use | Small and medium datasets | Very large corpora, sampling, audits |
| Persistence | Save full transformed dataset | Save bounded sample or report |

## Intended asset graph

```
dolma_stream
     |
     v
streaming_report
```

`dolma_stream` represents a bounded streamed sample. `streaming_report` records
the sample size and any aggregate metrics that can be computed without loading
the full source dataset.

## Key API

```python
@hf_dataset_asset(
    path="allenai/dolma",
    split="train",
    streaming=True,
    group_name="streaming_oom_datasets",
)
def dolma_stream(context, dataset) -> MaterializeResult:
    rows = list(islice(dataset, 1000))
    sample = Dataset.from_list(rows)

    return MaterializeResult(
        value=sample,
        metadata={
            "sample_rows": len(sample),
            "streaming": True,
        },
    )
```

The important rule is to treat streamed inputs as iterators. Do not call
`len(dataset)`, do not assume random access, and keep materialized outputs
bounded.

## Metadata to surface

| Key | Description |
|-----|-------------|
| `sample_rows` | Number of streamed examples retained locally |
| `streaming` | Confirms the source was read in streaming mode |
| `source_dataset` | Hub dataset identifier |
| `processing_limit` | Maximum number of rows consumed from the stream |

## Storage layout

```
.dagster_hf_storage/
└── dolma_stream/        # bounded sample, not the full corpus
```

`streaming_report` returns a `MaterializeResult` with report data and metadata.

## How to run

```bash
pip install dagster dagster-hf-datasets 

cd dagster_hf_datasets_examples

dagster dev -m streaming_oom_datasets.definitions
```

Materialize the streamed sample first, then the report. Keep the sample limit
small for local development and increase it only when the Dagster run worker has
enough memory and disk budget.
