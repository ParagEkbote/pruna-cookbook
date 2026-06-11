# Multimodal Dataset Profiling

Analyze an image-text dataset and generate automated quality insights:
resolution statistics, caption analysis, thumbnail gallery, and a
composite health score.

## What this example shows

- Accessing `PIL.Image` objects from a Hub dataset via `dataset["image"]`
- Computing per-image resolution, aspect ratio, and color mode statistics
- Analyzing caption token counts and vocabulary size across 5 captions per image
- Writing a thumbnail gallery to disk with a JSON manifest
- Combining image and caption signals into a health score with flagging logic
- Returning plain `dict` results for report-style assets that don't need IO manager persistence

## Dataset

[`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k) — 31,783 Flickr images each paired with
5 human-written captions. Stored with PIL Image objects in the `image`
column and caption lists in the `caption` column.

> **Note:** `nlphuji/flickr30k` requires a Hub login. Set `HF_TOKEN`
> or pass `token=` to `HuggingFaceResource`.

## Asset graph

```
flickr30k_raw
   /        \
image_stats  caption_stats   sample_gallery
         \       /
      dataset_health_report
```

## Key implementation details

**Accessing images:** PIL Images are returned directly from dataset iteration:
```python
for example in dataset:
    img = example["image"]   # PIL.Image.Image
    width, height = img.size
```

**Thumbnail generation:** `PIL.Image.thumbnail()` is in-place and
maintains aspect ratio:
```python
img = example["image"].copy()   # copy before mutating
img.thumbnail((128, 128))
img.save(out_path, format="JPEG")
```

**Health score formula:**
```
health_score = 100
  - (extreme_aspect_pct × 0.30)
  - (short_caption_pct  × 0.40)
  - (missing_caption_pct × 0.30)
```
Captions are weighted most heavily as they are the primary text signal.

## Flagged quality issues

| Issue | Threshold | Weight in score |
|-------|-----------|-----------------|
| Extreme aspect ratio | < 0.2 or > 5.0 | 30% |
| Short captions | avg < 4 tokens | 40% |
| Missing captions | empty list | 30% |

## Storage layout

```
.dagster_hf_storage/
├── flickr30k_raw/
├── image_stats/           # Dataset with per-image width/height/aspect/mode
├── caption_stats/         # Dataset with per-example caption length stats
└── sample_gallery/        # Written directly by asset (not via IO manager)
    ├── sample_0000.jpg
    ├── sample_0001.jpg
    ...
    ├── sample_0015.jpg
    └── manifest.json
```

`sample_gallery` and `dataset_health_report` return plain `dict` values
and are not persisted by the IO manager.

## How to run

```bash
pip install dagster dagster-hf-datasets Pillow
export HF_TOKEN=hf_...
dagster dev -f definitions.py
```

Materialize `flickr30k_raw` first, then `image_stats`, `caption_stats`,
and `sample_gallery` in parallel, then `dataset_health_report` last.