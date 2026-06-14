# Image Dataset Curation Pipeline

Build a production-quality vision dataset from a noisy image collection
through a multi-stage curation funnel with multi-format export.

## What this example shows

- Multi-stage curation pipeline: resolution → corruption → dedup → aspect ratio
- Two-pass corrupt image detection using `PIL.Image.verify()` + `img.load()`
- Perceptual deduplication via 32×32 RGB thumbnail hashing
- Aspect ratio validation with configurable thresholds
- Multi-format export: Arrow (`save_to_disk`) + Parquet (`to_parquet`) + JSON manifest
- Full funnel report tracking row counts dropped at every stage
- `@asset_check` with ERROR severity for hard bounds violations

## Dataset

[`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet) — 100,000 training images
across 200 ImageNet classes, resized to 64×64. Compact enough for local
runs while exercising all curation stages realistically.

| Split | Rows | Classes |
|-------|------|---------|
| train | 100,000 | 200 |
| valid | 10,000 | 200 |

## Asset graph

```
tiny_imagenet_raw
        │
        ▼
resolution_filtered         (drop < 32px or > 4096px)
        │
        ▼
corrupt_removed             (verify + load pass)
        │
        ▼
deduplicated_images         (32×32 RGB pixel hash)
        │
        ▼
aspect_ratio_validated      (keep 0.25 – 4.0)
      /   \
     ▼     ▼
curated_  curation_report   (full funnel stats)
export    (+ all upstream stages)
```

## Curation thresholds

| Stage | Rule | Default |
|-------|------|---------|
| Resolution | `MIN_WIDTH/HEIGHT = 32`, `MAX = 4096` | configurable at top of `assets.py` |
| Corruption | `img.load()` raises → corrupt | — |
| Deduplication | 32×32 RGB MD5 hash match | — |
| Aspect ratio | `0.25 ≤ w/h ≤ 4.0` | configurable |

## Asset checks

| Check | Severity | Condition |
|-------|----------|-----------|
| `check_curation_retention` | WARN | Curated set ≥ 90% of deduplicated rows |
| `check_aspect_bounds` | ERROR | Zero aspect ratio violations in final output |

## Storage layout

```
.dagster_hf_storage/
├── tiny_imagenet_raw/
├── resolution_filtered/
├── corrupt_removed/
├── deduplicated_images/
├── aspect_ratio_validated/
└── curated_images/
    ├── arrow/             ← save_to_disk() format
    ├── curated.parquet    ← portable Parquet
    └── manifest.json
```

## Corrupt detection note

PIL's `verify()` is destructive — it consumes the image file pointer.
Always call it on a `.copy()` and use `img.load()` as the primary check
for datasets where images are already decoded into memory:

```python
def _check_corrupt_via_load(img) -> bool:
    try:
        img.load()
        return False
    except Exception:
        return True
```

## How to run

```bash
pip install dagster dagster-hf-datasets Pillow
cd dagster_hf_datasets_examples

dagster dev -m image_dataset_curation.definitions
```

Materialize sequentially from `tiny_imagenet_raw` through `aspect_ratio_validated`,
then `curated_export` and `curation_report` in parallel as the final stage.