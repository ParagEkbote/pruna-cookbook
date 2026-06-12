# Preference Alignment Data

This example demonstrates **preference pair validation and preparation** for RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) training, featuring two modern alignment datasets.

## Datasets

### 1. Anthropic/hh-rlhf (Classic RLHF Data)

- **Size**: ~169K preference pairs
- **Structure**: Chosen vs. Rejected responses to prompts
- **Use Case**: Foundational RLHF dataset used to train Claude and Llama 2
- **Why It Matters**: Established standard; if your model can handle hh-rlhf, it's ready for production RLHF

**Asset**: `dpo_training_dataset`

### 2. allenai/ultrafeedback_binarized_cleaned (Modern High-Quality)

- **Size**: ~64K preference pairs
- **Structure**: (instruction, chosen, rejected) tuples
- **Quality**: Binarized + cleaned (higher quality than raw UltraFeedback)
- **Annotation**: Multi-model annotations provide diversity
- **Use Case**: DPO/ORPO training; better generalization than model-specific data

**Asset**: `ultrafeedback_preference_dataset`

## Why Both?

| Aspect | hh-rlhf | UltraFeedback |
|--------|---------|---------------|
| **Maturity** | Mature, proven | Newer, cutting-edge |
| **Quality** | Foundational | High-quality curated |
| **Model Specificity** | Anthropic-tuned | Model-agnostic |
| **Use Case** | RLHF baseline | DPO/ORPO alternative |
| **Size** | 169K pairs | 64K pairs |
| **Generalization** | Good for Claude-style | Better for diverse LLMs |

## Pipeline: Preference Pair Validation

```
hh-rlhf (raw)              UltraFeedback (raw)
    ↓                              ↓
validate (is_valid filter)     validate (is_valid filter)
    ↓                              ↓
dpo_training_dataset       ultrafeedback_preference_dataset
  (169K → ~160K)              (64K → ~60K)
    ↓                              ↓
Ready for DPO/ORPO training
```

## Assets

### 1. `dpo_training_dataset` → `MaterializeResult` (hh-rlhf)

**Validation Rules**:
- `chosen` and `rejected` must be non-null
- Both must be non-empty after stripping whitespace
- `chosen ≠ rejected` (cannot be identical)

**Metadata Output**:
```json
{
  "original_rows": 169000,
  "validated_rows": 160500,
  "removed_rows": 8500,
  "dataset": "Anthropic/hh-rlhf",
  "fingerprint": "abcd1234"
}
```

### 2. `ultrafeedback_preference_dataset` → `MaterializeResult`

**Validation Rules**:
- `instruction` must be non-empty (additional validation vs. hh-rlhf)
- `chosen` and `rejected` must be non-null and non-empty
- `chosen ≠ rejected`

**Metadata Output**:
```json
{
  "original_rows": 64000,
  "validated_rows": 61500,
  "removed_rows": 2500,
  "dataset": "allenai/ultrafeedback_binarized_cleaned",
  "quality_note": "High-quality, model-agnostic preference pairs"
}
```

## Patterns Demonstrated

### 1. **Preference Pair Validation**
- Validates structure for DPO compatibility
- Ensures chosen > rejected (preference relationship)
- Handles dataset-specific field differences

### 2. **Quality Metrics**
- Tracks removed rows (logging data loss)
- Computes retention % (data quality health)

### 3. **Multi-Dataset Support**
- Different datasets, same validation pattern
- Shows how to extend to other preference sources

## Running Locally

```bash
cd preference_alignment_data
dagster dev
```

Materialize both assets:
1. `dpo_training_dataset` (hh-rlhf)
2. `ultrafeedback_preference_dataset`

Compare metadata in Dagster UI to see quality differences.

## Use Cases

### Training DPO Models
```python
# After running this example
from datasets import load_from_disk

train_data = load_from_disk(".dagster_hf_storage/dpo_training_dataset")
# or
train_data = load_from_disk(".dagster_hf_storage/ultrafeedback_preference_dataset")

# Use with your DPO trainer
trainer = DPOTrainer(model=model, args=args, train_dataset=train_data)
```

### Mixing Datasets
```python
# Combine both for diverse training
from datasets import concatenate_datasets

combined = concatenate_datasets([
    load_from_disk(".dagster_hf_storage/dpo_training_dataset"),
    load_from_disk(".dagster_hf_storage/ultrafeedback_preference_dataset"),
])
```

## Customization

### Add More Validation Rules

```python
def is_valid(example):
    # Existing checks...
    if not is_valid(example):
        return False
    
    # Additional: length constraints
    if len(example["chosen"]) < 10:
        return False
    
    # Additional: reject low-quality formats
    if "unable to" in example["rejected"].lower():
        return False
    
    return True
```

### Add Downstream Processing

```python
@asset(group_name="preference_alignment")
def dpo_formatted_pairs(dpo_training_dataset: Dataset) -> Dataset:
    """Convert to OpenAI ChatML format for fine-tuning."""
    def to_chatml(example):
        return {
            "chosen": f"<|im_start|>assistant\n{example['chosen']}<|im_end|>",
            "rejected": f"<|im_start|>assistant\n{example['rejected']}<|im_end|>",
        }
    return dpo_training_dataset.map(to_chatml)
```

### Add Quality Scoring

```python
@asset(group_name="preference_alignment")
def preference_quality_scores(ultrafeedback_preference_dataset: Dataset) -> dict:
    """Compute agreement metrics between chosen/rejected."""
    # Compute length ratios, diversity, etc.
    return quality_report
```

## References

- [hh-rlhf Paper](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [UltraFeedback Paper](https://arxiv.org/abs/2310.01852)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Related Examples](../):
  - `multi_modal_data_profiling/` — Vision-language preferences
  - `dataset_card_publishing/` — Publishing aligned datasets to Hub

## Tips

- **For RLHF Baselines**: Start with `dpo_training_dataset` (hh-rlhf)
- **For ORPO/DPO**: Try `ultrafeedback_preference_dataset` (better diversity)
- **For Large-Scale**: Combine both + add your own data collection pipeline
- **For Evaluation**: Use preference pairs to build LLM evaluation sets
