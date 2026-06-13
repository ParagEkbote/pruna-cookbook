# Audio Dataset Curation

Curate high-quality speech clips from the Common Voice dataset by validating duration,
audio format, and language tags.

## What this example shows

- Loading multimodal audio datasets with `@hf_dataset_asset`
- Validating audio objects: checking array structure and duration ranges
- Filtering by language locale tags
- Quantifying dataset retention through materialization metadata
- Processing audio fields with nested dict structures (`audio["array"]`, `audio["duration"]`)

## Dataset

[`mozilla-foundation/common_voice_11_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) (English split) — 240K+
crowdsourced speech recordings across diverse speakers, ages, accents, and recording conditions.
High-quality audio resource for training speech recognition and speech-to-text models.

| Metric | Value |
|--------|-------|
| **Language** | English (en) |
| **Total clips** | ~240K |
| **Duration range** | 2–6 seconds (typical) |
| **Audio format** | PCM, 48kHz |

## Curation rules

| Rule | Threshold | Reason |
|------|-----------|--------|
| **Duration** | 1–20 seconds | Filter too-short (noise) and too-long (out-of-spec) clips |
| **Audio array** | Must exist and be non-null | Reject corrupted/malformed samples |
| **Language tag** | Must start with `en` | English-only filtering |

## Key API

```python
@hf_dataset_asset(
    path="mozilla-foundation/common_voice_11_0",
    config="en",
    split="train",
)
def curated_audio_dataset(context, dataset: Dataset) -> MaterializeResult:
    def is_valid(example):
        audio = example.get("audio")
        if audio is None or "array" not in audio:
            return False
        if audio.get("duration", 0) < MIN_DURATION or > MAX_DURATION:
            return False
        return example.get("locale", "").startswith("en")

    curated = dataset.filter(is_valid)
    return MaterializeResult(value=curated, metadata={
        "original_rows": len(dataset),
        "curated_rows": len(curated),
        "retention_pct": round(100 * len(curated) / len(dataset), 1),
    })
```

## Metadata returned

| Key | Description |
|-----|-------------|
| `original_rows` | Row count before filtering |
| `curated_rows` | Row count after filtering |
| `removed_rows` | Count of invalid samples dropped |
| `dataset` | Source dataset identifier |
| `fingerprint` | Reproducibility hash from the datasets library |

## Storage layout

```
.dagster_hf_storage/
└── curated_audio_dataset/     # Arrow format via save_to_disk()
```

## How to run

```bash
pip install dagster dagster-hf-datasets
dagster dev -f definitions.py
```

Then navigate to the **Asset Catalog** in the Dagster UI
([http://localhost:3000](http://localhost:3000)) and materialize `curated_audio_dataset`.
