# Vision-Language Scale Pipeline

This example demonstrates **large-scale multimodal data processing** with deduplication and quality filtering using [LAION-COCO](https://huggingface.co/datasets/laion/LAION-COCO), a 600M image-text pair dataset designed for vision-language model training.

## Dataset: LAION-COCO

[LAION-COCO](https://huggingface.co/datasets/laion/LAION-COCO) is a curated subset of LAION (Large-scale Artificial Intelligence Open Network), optimized for vision-language model instruction-tuning:

- **Size**: ~600M high-quality image-text pairs
- **Curation**: Filtered from LAION for better caption-image alignment
- **Quality**: Balanced between scale and quality (unlike raw LAION which can be noisy)
- **Use Cases**: Training CLIP alternatives, vision-language instruction-tuned models (LLaVA, Qwen-VL, Flamingo alternatives)


## Pipeline Architecture

Multi-stage filtering and deduplication workflow:

```
raw_laion_coco (5K sample; ~600M at scale)
    ↓
caption_quality_filtered (Length: 3-500 words; no boilerplate)
    ↓
deduplicated_by_caption_hash (SHA256 dedup; remove exact duplicates)
    ↓
language_identified_captions (Detect language; filter to English)
    ↓
dedup_quality_report (Compute retention %, statistics, quality score)
```

## Key Assets

### 1. **raw_laion_coco** → `MaterializeResult`

Ingests LAION-COCO (600M pairs, sampled to 5K for dev):
- Analyzes caption characteristics (length distribution, sample captions)
- Metadata: row count, total dataset size, sample captions

**Output**:
```json
{
  "rows": 5000,
  "total_dataset_size": 600000000,
  "avg_caption_tokens": 18.5,
  "sample_captions": ["A dog running in a park", "...", "..."],
  "modality": "image-text"
}
```

### 2. **caption_quality_filtered** → `MaterializeResult`

**Filtering rules**:
- **Minimum**: 3 words (skip single-word or 2-word captions)
- **Maximum**: 500 words (skip rambling/off-topic captions)
- **No boilerplate**: Exclude "website", "banner", "advertisement", "click here", etc.

**Retention Typical**: 95-98% (most captions are already reasonably good quality)

**Metrics Logged**:
- Input rows, output rows, retention %

**Pattern Reused**: Similar to `sanitization_observability` — multi-rule quality filtering at scale

### 3. **deduplicated_by_caption_hash** → `MaterializeResult`

Removes exact duplicate captions using SHA256 hashing:
- **Why**: At 600M scale, even 1% duplicates = 6M redundant pairs
- **How**: Hash each caption text; track seen hashes; skip duplicates
- **Efficiency**: O(N) time, O(unique captions) memory
- **Output**: Deduplicated dataset + duplicate statistics

**Metrics Logged**:
- Duplicates removed (count and %)
- Retention % post-dedup

**Use Case Example**: Multiple images might have identical caption ("a dog"; "golden retriever"; etc.) — dedup catches these.

### 4. **language_identified_captions** → `MaterializeResult`

Identifies caption language and filters to English:
- **Detection**: Simple ASCII-ratio heuristic (production: use `langdetect` or `textblob`)
- **Rule**: > 90% ASCII = English, 70-90% = Mixed, < 70% = Non-English
- **Customization**: Can modify to keep multilingual or detect specific languages

**Retention Typical**: 70-85% (LAION-COCO is diverse; includes non-English captions)

**Metrics Logged**:
- English pairs, non-English pairs, retention %

### 5. **dedup_quality_report** → `dict` (report)

Comprehensive pipeline metrics across all stages:

```json
{
  "pipeline_stage": "complete",
  "raw_pairs": 5000,
  "after_quality_filter": 4950,
  "after_deduplication": 4870,
  "after_language_filter": 3920,
  "total_retention_pct": 78.4,
  "quality_filter_retention_pct": 99.0,
  "deduplication_retention_pct": 98.4,
  "language_filter_retention_pct": 80.4,
  "caption_length_stats": {
    "mean_tokens": 17.8,
    "median_tokens": 16.0,
    "min_tokens": 3,
    "max_tokens": 500
  },
  "data_quality_score": 62.7
}
```

**Quality Score Formula**: `retention_pct × 0.8`
- Lower scores reflect filtering intensity
- Scores 60-80% typical for production pipelines

## Patterns Demonstrated

### 1. **Multi-Stage Filtering at Scale**
- Quality filter → Deduplication → Language filtering
- Each stage reduces dataset size but improves quality
- Pattern reusable for other large-scale datasets (text, audio, etc.)

### 2. **Hash-Based Deduplication**
- Memory-efficient O(unique captions) memory usage
- Detects exact duplicates only (not fuzzy/near-duplicates)
- Scales to billions of records

### 3. **Language Detection**
- Simple heuristics (ASCII ratio) for speed
- Can integrate pretrained language models for accuracy
- Demonstrates trade-off between speed and precision

### 4. **Metadata Tracking Through Pipeline**
- Logs retention % at each stage
- Enables debugging (where did data get lost?)
- Pattern from `sanitization_observability` applied to multimodal

## Running Locally

```bash
cd vision_language_scale_pipeline
dagster dev
```

Materialize order:
1. `raw_laion_coco` (sample and analyze)
2. `caption_quality_filtered` (filter by length/content)
3. `deduplicated_by_caption_hash` (remove exact duplicates)
4. `language_identified_captions` (language filtering)
5. `dedup_quality_report` (final metrics)

**Note**: First run downloads LAION-COCO sample (~100-200MB). Uses cache thereafter.

## Customization

### Adjust Sample Size

```python
# In raw_laion_coco()
sample_size = min(50000, len(dataset))  # Larger sample for better statistics
```

### Keep Multilingual Data

```python
# In language_identified_captions(), remove filter:
# Just return dataset as-is or add language tag
```

### Implement Fuzzy Deduplication

```python
from difflib import SequenceMatcher

def is_duplicate(caption1, caption2, threshold=0.95):
    ratio = SequenceMatcher(None, caption1, caption2).ratio()
    return ratio > threshold

# In deduplicated_by_caption_hash():
# Compare against previously seen captions using fuzzy matching
```

### Add Image Validation

```python
@asset
def image_url_validated(language_identified_captions: Dataset) -> Dataset:
    """Validate image URLs are reachable (optional, slow)."""
    import requests
    
    def url_exists(example):
        url = example.get("url")
        try:
            response = requests.head(url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    return language_identified_captions.filter(url_exists)
```

### Add Caption Length Constraints

```python
# Existing: 3-500 word range
# Add: Min/max character constraints

def is_quality_caption(example):
    caption = example.get("caption", "").strip()
    words = caption.split()
    
    # Existing checks...
    if len(words) < 3 or len(words) > 500:
        return False
    
    # New: character-level constraints
    if len(caption) < 20:  # Too short in chars
        return False
    if len(caption) > 5000:  # Too long in chars
        return False
    
    return True
```

## Use Cases

### Training Vision-Language Models
- Use deduplicated, high-quality LAION-COCO to train CLIP alternatives
- Instruction-tune LLaVA or similar VLMs

### Building Search Engines
- Use image-caption pairs to train dual-encoder retrieval models
- Index images for text-to-image search

### Data Quality Auditing
- Monitor pipeline metrics to detect dataset drift
- Compare quality reports across time periods

### Mixed Dataset Training
Combine with other VL datasets:
```python
@asset
def multi_dataset_blend(
    laion_coco_clean: Dataset,
    other_vl_dataset: Dataset,
) -> Dataset:
    """Blend LAION-COCO with custom or proprietary VL data."""
    from datasets import concatenate_datasets
    return concatenate_datasets([laion_coco_clean, other_vl_dataset])
```

## Performance Notes

| Stage | Time (10K pairs) | Memory |
|-------|-----------------|--------|
| Load raw | ~5 sec | 200MB |
| Quality filter | ~2 sec | 50MB |
| Deduplication | ~15 sec | 100MB (hashes) |
| Language filter | ~3 sec | 50MB |
| Reporting | ~5 sec | 100MB |
| **Total** | ~30 sec | ~500MB |

**At 600M scale**: ~8 hours (parallelizable per language or partition)

## See Also

- [LAION-COCO on Hub](https://huggingface.co/datasets/laion/LAION-COCO)
- [LAION Project](https://laion.ai/)
- [Related Examples](../):
  - `multi_modal_data_profiling/` — Image-specific statistics
  - `sanitization_observability/` — Advanced quality metrics
  - `code_instruction_pipeline/` — Language-specific filtering (for code)

## Tips

- **For CLIP training**: Use full pipeline (quality + dedup + language filter)
- **For VQA**: Keep multilingual data or filter to diverse languages
- **For Evaluation**: Use final deduplicated set as quality benchmark
- **For Production**: Add image URL validation + OCR-based quality scoring
