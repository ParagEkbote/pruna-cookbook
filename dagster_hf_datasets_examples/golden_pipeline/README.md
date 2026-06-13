# Golden Pipeline: Multi-Stage Web Data Cleaning

This example demonstrates a **multi-stage data cleaning pipeline** using the modern **FineWeb** dataset, showcasing best practices for preparing web-scale text data for LLM pretraining.

## Dataset: FineWeb

[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is a high-quality, deduplicated web corpus containing **15 trillion tokens** of text across diverse domains.

## Pipeline Architecture

The pipeline demonstrates a **multi-stage cleaning workflow**:

```
fineweb_raw (Sample 10K rows)
    ↓
fineweb_cleaned (Filter: text length > 50 chars)
    ↓
fineweb_quality_validated (Verify non-null text)
    ↓
quality_report (Compute metrics)
    ├→ fineweb_train (90% train split)
    │   ↓
    │   fineweb_train_tokenized
    │       ↓
    │       dataset_card
    │           ↓
    │           hub_publication_manifest
    │
    └→ fineweb_test (10% test split)
        ↓
        fineweb_test_tokenized
```

## Key Assets

### 1. **fineweb_raw** → `MaterializeResult`
- Ingests FineWeb sample (10K rows for dev)
- Metadata: row count, source dataset, config

### 2. **fineweb_cleaned** → `Dataset`
- **Filtering Rule**: Text must be > 50 characters after stripping whitespace
- Logs retention % (typically 80-95% for web data)
- Demonstrates simple quality filtering

### 3. **fineweb_quality_validated** → `Dataset`
- **Validation**: Ensures text field is non-null
- Handles edge cases from upstream processing

### 4. **quality_report** → `dict`
- Aggregates metrics from validated data
- Logged to Dagster UI for visibility

### 5. **fineweb_train / fineweb_test** → `Dataset`
- 90/10 train-test split with fixed seed (reproducibility)
- Downstream assets for tokenization

### 6. **fineweb_train_tokenized / fineweb_test_tokenized** → `Dataset`
- Tokenizes text using `bert-base-uncased`
- Demonstrates batch processing with HuggingFace `map()`

### 7. **dataset_card** → `str`
- Generates a markdown summary
- Used for Hub publication workflows

### 8. **hub_publication_manifest** → `dict`
- Final metadata for publishing to Hub
- Demonstrates publication readiness pattern

## Patterns Demonstrated

### Multi-Stage Filtering
- **Stage 1**: Length-based quality (> 50 chars)
- **Stage 2**: Null checks
- **Stage 3**: Tokenization validation

→ *Real-world datasets often require 5+ filtering stages; this shows the pattern*

### Train-Test Splitting
- Fixed seed (42) for reproducibility
- Stratified splits preserve data distribution

### Metadata Tracking
- Each stage logs metrics to UI
- Enables debugging and monitoring

### Downstream Tokenization
- Shows integration with transformers library
- Pattern reusable for other tokenizers (GPT-2, Llama, etc.)

## Running Locally

```bash
cd golden_pipeline
dagster dev
```

Then open http://localhost:3000 and materialize the assets.

**Note**: First run may take a few minutes to download FineWeb (even 10K sample is ~5MB).

## Customization

### Change Dataset Size
In `fineweb_raw()`, modify the sample size:
```python
dataset.select(range(min(50000, len(dataset))))  # Larger sample
```

### Change Tokenizer
In `assets.py`, update `TOKENIZER`:
```python
TOKENIZER = "mistralai/Mistral-7B"  # Or any HuggingFace model
```

### Add More Filtering Stages
Create new assets with additional filters:
```python
@asset(group_name="golden_master_pipeline")
def fineweb_deduped(fineweb_quality_validated: Dataset) -> Dataset:
    """Remove near-duplicate captions using fuzzy matching."""
    # Your deduplication logic
    return deduped_dataset
```

## Integration with Dagster Sensors

To automatically refresh when FineWeb is updated on the Hub:
```python
from dagster_hf_datasets import hf_dataset_sensor

@hf_dataset_sensor(path="HuggingFaceFW/fineweb")
def fineweb_updated(context):
    """Trigger pipeline on dataset revision change."""
```

## See Also

- [FineWeb on Hub](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb Paper](https://huggingface.co/papers/2406.04333)
- Related examples:
  - `distributed_token_sharding/` — Large-scale token partitioning
  - `sanitization_observability/` — Advanced data quality metrics
