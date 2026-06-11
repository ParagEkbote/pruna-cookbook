# Automated Dataset Card Publishing

Generate and push a Hub dataset card directly from Dagster pipeline
metadata using `HFDatasetPublisher`.

## What this example shows

- Using `HFDatasetPublisher` to programmatically generate a Hub README
- Embedding run metadata (Dagster run ID, timestamps, processing steps) into the card
- Wiring pipeline lineage (source dataset, transformations, row counts) into card content
- Dry-run mode: generating card content without publishing (safe for local dev)
- Publishing to the Hub with `publish()` when `HF_TOKEN` is available

## Dataset

[`rajpurkar/squad`](https://huggingface.co/datasets/rajpurkar/squad) — Stanford Question Answering Dataset.
100K+ reading comprehension QA pairs over Wikipedia passages. Ships a
well-structured existing card, making it a useful reference for
card content patterns.

| Split | Rows |
|-------|------|
| train | 87,599 |
| validation | 10,570 |

## Key API

```python
# NOTE: HFDatasetPublisher is in a private module — pin your version.
from dagster_hf_datasets._export._publisher import HFDatasetPublisher

publisher = HFDatasetPublisher(
    repo_id="your-username/squad-enriched",
    source_dataset="rajpurkar/squad",
    source_revision=dataset._fingerprint,
    processing_steps=[...],
    metadata={...},
)

# Dry run — returns card markdown without pushing
card_content = publisher.generate_card()

# Live publish — requires HF_TOKEN with write access
hub_url = publisher.publish(dataset=enriched_dataset)
```

## Asset graph

```
squad_train
     │
     ▼
squad_enriched          (adds answer_length column)
     │
     ▼
squad_dataset_card      (generates + optionally publishes README)
```

## Processing steps documented in the card

| Step | Description |
|------|-------------|
| `ingestion` | Load from `rajpurkar/squad` train split |
| `enrichment` | Add `answer_length` column (token count of first answer span) |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For publishing | Hub token with write access to `HF_REPO_ID` |
| `HF_REPO_ID` | For publishing | Target repo, e.g. `your-username/squad-enriched` |

The example runs in **dry-run mode** by default (card generated, not pushed).
To enable publishing, uncomment the `publisher.publish()` call in `assets.py`.

## Storage layout

```
.dagster_hf_storage/
├── squad_train/
└── squad_enriched/
```

`squad_dataset_card` returns a plain `dict` and is not persisted by the IO manager.

## How to run

```bash
pip install dagster dagster-hf-datasets
# Optional: export HF_TOKEN=hf_... HF_REPO_ID=your-username/squad-enriched
dagster dev -f definitions.py
```

Materialize in order: `squad_train` → `squad_enriched` → `squad_dataset_card`.