# Dataset Sanitization & Observability

Clean and validate a noisy web-crawl dataset while exposing quality
metrics as structured metadata in the Dagster UI.

## What this example shows

- Chaining `@hf_dataset_asset` with downstream `@asset` nodes for multi-step cleaning
- Filtering null/empty/short text rows using `Dataset.filter()`
- Deduplication via prefix hashing (first 500 chars → MD5)
- `@asset_check` for post-cleaning validation with `ERROR` and `WARN` severities
- A dedicated quality report asset that emits structured metadata visible in the asset catalog

## Dataset

[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT` config) — a 10-billion-token
sample of web-crawl documents scored for educational quality. Real-world noise
(short stubs, near-duplicates, malformed entries) makes it well-suited for
demonstrating sanitization pipelines.

> **Note:** The `sample-10BT` config is large. For local development, stream a
> small slice first or set `streaming=True` on the asset and adjust the pipeline
> accordingly (see example 10 for the streaming pattern).

## Asset graph

```
raw_fineweb_edu
      │
      ▼
filtered_fineweb_edu        (drop null / short rows)
      │
      ▼
deduplicated_fineweb_edu    (drop prefix-hash duplicates)
      │           │
      ▼           ▼
 [checks]   cleaning_quality_report
```

## Key API

```python
# Ingest via decorator — function body receives injected dataset
@hf_dataset_asset(path="HuggingFaceFW/fineweb-edu", config="sample-10BT", split="train")
def raw_fineweb_edu(context, dataset: Dataset) -> MaterializeResult: ...

# Downstream transformation — plain @asset, dataset flows via IO manager
@asset
def filtered_fineweb_edu(raw_fineweb_edu: Dataset) -> Dataset: ...

# Asset check — validates the cleaned output
@asset_check(asset=deduplicated_fineweb_edu)
def check_no_null_text(deduplicated_fineweb_edu: Dataset) -> AssetCheckResult: ...
```

## Asset checks

| Check | Severity | Condition |
|-------|----------|-----------|
| `check_no_null_text` | ERROR | Zero null/empty text rows after deduplication |
| `check_retention_rate` | WARN | Cleaned dataset retains ≥ 80% of raw rows |

## Storage layout

```
.dagster_hf_storage/
├── raw_fineweb_edu/
├── filtered_fineweb_edu/
└── deduplicated_fineweb_edu/
```

`cleaning_quality_report` returns a plain `dict` and is not persisted by the IO manager.

## Metadata visible in the Dagster UI

| Asset | Key | Description |
|-------|-----|-------------|
| `raw_fineweb_edu` | `null_text_count` | Null/empty text rows in raw data |
| `raw_fineweb_edu` | `short_text_count` | Rows with < 10 tokens |
| `cleaning_quality_report` | `retention_pct` | % of rows surviving the full pipeline |
| `cleaning_quality_report` | `dropped_rows` | Total rows removed |

## How to run

```bash
pip install dagster dagster-hf-datasets
dagster dev -f definitions.py
```

Materialize assets in order: `raw_fineweb_edu` → `filtered_fineweb_edu` →
`deduplicated_fineweb_edu` → `cleaning_quality_report`. Then run asset checks
from the **Checks** tab on `deduplicated_fineweb_edu`.