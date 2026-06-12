# 13 · Synthetic Multimodal Data Generation (CPU)

Build a synthetic image-caption dataset by captioning pre-generated
Stable Diffusion images and scoring how well the generated caption
aligns with the original prompt — entirely on CPU.

## What this example shows

- Using an existing Hub dataset of AI-generated images as a synthetic data source (no GPU inference)
- Free quality filtering using a dataset's precomputed metadata (`image_nsfw`, `prompt_nsfw`)
- CPU-based image captioning with `transformers` (BLIP-base)
- CPU-based semantic similarity scoring with `sentence-transformers` (MiniLM)
- Lazy global model loading to avoid reloading per-row
- A multi-stage funnel report and asset checks on the final filtered output

## Dataset

[`poloclub/diffusiondb`](https://huggingface.co/datasets/poloclub/diffusiondb) (`2m_random_1k` config) — 1,000
random (image, prompt, generation-parameter) triples from Stable
Diffusion generations submitted to the DiffusionDB Discord. Only
`SAMPLE_SIZE` (default **40**) rows are used to keep CPU runtime short.

| Field | Description |
|-------|-------------|
| `image` | Generated image (PIL) |
| `prompt` | Text prompt used to generate it |
| `seed`, `cfg`, `sampler`, `step` | Generation parameters |
| `image_nsfw`, `prompt_nsfw` | Precomputed NSFW scores (0-1, or -1 if unscored) |

## Asset graph

```
diffusiondb_sample
       │
       ▼
nsfw_filtered              (drop image_nsfw/prompt_nsfw >= 0.5)
       │
       ▼
synthetic_captions         (BLIP-base captions each image, CPU)
       │
       ▼
caption_alignment_scores   (MiniLM cosine similarity: prompt vs. caption)
       │
       ▼
synthetic_dataset_final    (keep alignment_score >= 0.2)
       │
       ▼
synthetic_generation_report
       │
   [checks]
```

## Key implementation details

**Lazy global model loading** — avoids reloading BLIP/MiniLM on every
row by caching at module scope:

```python
_blip_model = None

def _load_blip():
    global _blip_model
    if _blip_model is None:
        _blip_model = BlipForConditionalGeneration.from_pretrained(...)
    return _blip_model
```

**Per-image captioning loop** (no batching — keeps CPU memory bounded):

```python
for example in nsfw_filtered:
    inputs = processor(example["image"].convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
```

**Alignment scoring** via sentence-embedding cosine similarity:

```python
emb = model.encode([prompt, generated_caption], convert_to_tensor=True)
score = float(util.cos_sim(emb[0], emb[1]))
```

## Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `SAMPLE_SIZE` | 40 | Rows processed; controls CPU runtime |
| `NSFW_THRESHOLD` | 0.5 | Drop rows scoring above this on image/prompt NSFW |
| `ALIGNMENT_THRESHOLD` | 0.2 | Minimum caption↔prompt cosine similarity to keep |

All three are tunable constants at the top of `assets.py`.

## Asset checks

| Check | Severity | Condition |
|-------|----------|-----------|
| `check_no_empty_captions` | ERROR | No empty `generated_caption` values in final output |
| `check_mean_alignment` | WARN | Mean `alignment_score` in final dataset ≥ `ALIGNMENT_THRESHOLD` |

## Storage layout

```
.dagster_hf_storage/
├── diffusiondb_sample/
├── nsfw_filtered/
├── synthetic_captions/
├── caption_alignment_scores/
└── synthetic_dataset_final/
```

`synthetic_generation_report` returns a plain `dict` and is not persisted.

## How to run

```bash
pip install dagster dagster-hf-datasets Pillow
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers

dagster dev -f definitions.py
```

First run downloads BLIP-base (~990MB) and MiniLM (~90MB) to the local
HF cache — subsequent runs reuse the cache. With `SAMPLE_SIZE=40`,
expect the full pipeline to complete in roughly 1-3 minutes on a
modern CPU.

Materialize sequentially from `diffusiondb_sample` through
`synthetic_dataset_final`, then `synthetic_generation_report`, then
run checks from the **Checks** tab on `synthetic_dataset_final`.