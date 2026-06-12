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

---

## LLaVA-150K: Modern Instruction-Tuned Vision-Language Data

This example now includes **LLaVA-150K** alongside Flickr30K, showcasing modern instruction-tuning data for vision-language models.

### Why LLaVA?

| Aspect | Flickr30K | LLaVA-150K |
|--------|-----------|-----------|
| **Data Type** | Raw image-caption pairs | Instruction-response pairs (Q&A) |
| **Use Case** | Image captioning pretraining | Instruction-tuning VLMs |
| **Structure** | 5 captions per image | Instruction + response dialogue |
| **Size** | 31K images | 150K examples |
| **Modernity** | Classic benchmark (2014) | Modern instruction-tuning (2023+) |
| **Model Fit** | Generic image understanding | Following visual instructions |

### LLaVA Asset Graph

```
llava_instruct_raw (150K examples, sampled to 5K)
       |
llava_instruction_stats (analyze Q&A structure)
       |
llava_quality_profile (instruction-tuning quality metrics)
```

### Key Assets

#### 1. `llava_instruct_raw` → `MaterializeResult`

Ingests [liuhaotian/llava-instruct-150k](https://huggingface.co/datasets/liuhaotian/llava-instruct-150k):
- 150K image-instruction-response triplets
- Sampled to 5K for dev (or adjust as needed)
- Metadata: row count, column names, dataset info

**Columns**:
```python
{
    "image": PIL.Image,
    "conversations": [
        {"from": "human", "value": "What is in the image?"},
        {"from": "gpt", "value": "The image shows a dog..."}
    ]
}
```

#### 2. `llava_instruction_stats` → `Dataset`

Analyzes instruction-response pair structure:
- **Instruction tokens**: How complex are the questions?
- **Response tokens**: How detailed are the answers?
- **Is question**: Binary flag (ends with `?`)
- Metrics logged: token distributions, question %, response diversity

**Output per example**:
```json
{
  "idx": 123,
  "instruction_tokens": 12,
  "response_tokens": 45,
  "instruction_length": 142,
  "response_length": 520,
  "is_question": true
}
```

#### 3. `llava_quality_profile` → `dict`

Computes quality metrics for instruction-tuning:
```json
{
  "total_examples": 5000,
  "valid_instruction_response_pairs": 4950,
  "very_short_responses_count": 50,
  "very_long_responses_count": 120,
  "balanced_responses": 4830,
  "balanced_response_pct": 96.6,
  "question_percentage": 68.5,
  "instruction_complexity_score": 1.2
}
```

**Quality Thresholds**:
- ✅ **Balanced response**: 5–500 tokens
- ❌ **Very short**: < 5 tokens (incomplete answers)
- ❌ **Very long**: > 500 tokens (off-topic rambling)

### Updated Asset Graph

```
flickr30k_raw               llava_instruct_raw
   /    |    \              |
  /     |     \             |
image_stats  caption_stats  llava_instruction_stats
  \      |     /            |
   \ health_report     llava_quality_profile
    \   /
    Both available in Dagster UI
```

### Comparing Image-Caption vs. Instruction-Tuned Data

| Metric | Flickr30K | LLaVA |
|--------|-----------|-------|
| Images per dataset | 31.8K | 150K |
| Text type | Captions (descriptive) | Instructions + Responses (interactive) |
| Caption avg length | ~15 tokens | Instruction: ~12 tokens, Response: ~45 tokens |
| Primary use | Image description pretraining | Visual QA & instruction-following |

### Running Both

```bash
dagster dev -f definitions.py
```

Materialize order:
1. `flickr30k_raw` + `llava_instruct_raw` (in parallel)
2. `image_stats` + `caption_stats` + `llava_instruction_stats` (in parallel)
3. `sample_gallery` + `dataset_health_report` + `llava_quality_profile` (in parallel)

### Use Cases

**Flickr30K**: Image-to-text pretraining, image understanding, captioning models

**LLaVA**: 
- Instruction-tuning VLMs (LLaVA, Qwen-VL, etc.)
- Visual question answering (VQA) 
- Building custom instruction-tuned vision-language models

### Customization

**Adjust LLaVA sample size**:
```python
# In llava_instruct_raw()
dataset.select(range(min(20000, len(dataset))))  # Larger sample
```

**Add your own instruction-tuned dataset**:
```python
@hf_dataset_asset(
    path="your-org/your-vl-dataset",
    split="train",
    group_name="multimodal_profiling",
)
def custom_instruct_raw(dataset: Dataset) -> MaterializeResult:
    ...
```

**Combine both datasets for joint training**:
```python
@asset(group_name="multimodal_profiling")
def combined_multimodal_data(
    flickr30k_raw: Dataset,
    llava_instruct_raw: Dataset,
) -> Dataset:
    """Mix image-caption and instruction-tuned data."""
    # Convert Flickr30K to instruction format
    # Concatenate with LLaVA
    # Return combined dataset
```