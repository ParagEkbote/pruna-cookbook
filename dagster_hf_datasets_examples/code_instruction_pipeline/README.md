# Code Instruction Pipeline

This example demonstrates **language-specific filtering and code-to-instruction conversion** for code generation LLM instruction-tuning, using the massive [BigCode the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) dataset.

## Dataset: BigCode the-stack-dedup

[The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) is a deduplicated collection of **3.1 billion files** across **358 programming languages**, sourced from public Git repositories. It was used to train:
- **StarCoder** (BigCode collaboration)
- **CodeLlama** (Meta)
- **Codestral** (Mistral AI)

**Why the-stack-dedup?**
- World's largest deduplicated code corpus
- Diverse language coverage (358 languages)
- Deduplicated to remove redundant files
- Foundation for modern code LLMs

> **Access note:** `bigcode/the-stack-dedup` is gated on the Hugging Face Hub.
> Accept the dataset terms on the Hub and set `HF_TOKEN` before running this
> example.

## Pipeline Architecture

The pipeline demonstrates **language-specific filtering and instruction extraction**:

```
raw_code_stack (10K sample across 358 languages)
    ↓
language_filtered_code (Filter to Python, JS, Go, Java, Rust, TypeScript, C++)
    ↓
instruction_examples (Convert code → instruction-response pairs)
    ↓
code_quality_metrics (Compute retention %, language distribution, quality scores)
```

## Key Assets

### 1. **raw_code_stack** → `MaterializeResult`

- Ingests BigCode the-stack-dedup sample (10K files for dev, adjust as needed)
- Analyzes language distribution in sample
- Metadata: row count, top 10 languages, total dataset size

**Output**:
```json
{
  "rows": 10000,
  "total_dataset_size": 3100000000,
  "languages_in_sample": 200+,
  "top_languages": {
    "Python": 2500,
    "JavaScript": 1800,
    "Java": 1200,
    "Go": 900,
    "TypeScript": 850
  }
}
```

### 2. **language_filtered_code** → `MaterializeResult`

**Filtering**: Keep only high-value, production-ready languages:
- Python (most popular for ML/AI)
- JavaScript (frontend + backend)
- Go (systems programming)
- Java (enterprise)
- Rust (performance-critical)
- TypeScript (modern web)
- C++ (low-level)

**Metrics Logged**:
- Input rows: 10,000
- Output rows: ~7,000-8,000 (70-80% retention typical)
- Retention %: Logged to UI for visibility

**Pattern Reused**: Similar to `dynamic_bucket_partitioning` — filtering by categorical attribute (language instead of spoken language)

### 3. **instruction_examples** → `MaterializeResult`

Converts code files into instruction-response pairs:
- **Instruction**: `"Write {language} code that solves the following problem."`
- **Response**: The actual code content
- **Metadata**: Language, code length, token count

**Output per example**:
```json
{
  "instruction": "Write Python code that solves the following problem.",
  "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    ...",
  "language": "Python",
  "code_length": 523,
  "token_count": 98
}
```

**Note**: This example treats entire files as responses. In production, you'd parse function/class definitions via AST to extract granular instruction-response pairs.

**Metrics Logged**:
- Instruction count extracted
- Average tokens per instruction
- Min/max token ranges

### 4. **code_quality_metrics** → `MaterializeResult` (report)

Aggregates metrics across pipeline stages:

```json
{
  "raw_files": 10000,
  "after_language_filter": 7500,
  "language_filter_retention_pct": 75.0,
  "instruction_examples": 7200,
  "instruction_extraction_rate": 96.0,
  "top_5_languages_post_filter": {
    "Python": 2000,
    "JavaScript": 1500,
    "Java": 1200,
    "Go": 900,
    "TypeScript": 900
  },
  "quality_score": 72.0
}
```

**Quality Score Formula**: `(instruction_examples / raw_files) × 100`
- Captures end-to-end retention (how much usable data survives filtering)
- Higher scores = more high-quality data available

## Patterns Demonstrated

### 1. **Language-Specific Filtering**
- Filters large corpus by categorical attribute (programming language)
- Pattern reusable for other categorizations (framework, library, file type)
- Matches pattern from `dynamic_bucket_partitioning/` (language → code language instead of natural language)

### 2. **Code-to-Instruction Conversion**
- Transforms raw code into instruction-response format
- Suitable for instruction-tuning code LLMs
- Demonstrates dataset transformation + format normalization

### 3. **Multi-Stage Metrics**
- Tracks retention at each pipeline stage
- Computes quality indicators for visibility
- Pattern reusable for other multi-stage pipelines

### 4. **Large-Scale Data Handling**
- Shows how to sample from 3.1B file dataset for development
- Patterns scale to full dataset with minor config changes

## Running Locally

```bash
cd code_instruction_pipeline
export HF_TOKEN=hf_...
dagster dev
```

Materialize order:
1. `raw_code_stack` (analyze language distribution)
2. `language_filtered_code` (filter to target languages)
3. `instruction_examples` (convert to instruction format)
4. `code_quality_metrics` (compute metrics)

**Note**: First run downloads the sample from BigCode Hub (~100-200MB). Subsequent runs use cached data.

## Customization

### Change Target Languages

```python
target_languages = {
    "Python", "JavaScript", "Go", "Java", "Rust",
    # Add your preferred languages:
    "Kotlin", "Swift", "Ruby", "PHP", "Scala"
}
```

### Adjust Sample Size

```python
# In raw_code_stack()
sample_size = min(50000, len(dataset))  # Larger sample for better statistics
```

### Improve Instruction Extraction

**Parse function definitions via AST**:
```python
import ast

def extract_functions(code, language):
    if language == "Python":
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function signature + docstring
                    yield {
                        "instruction": f"Implement: {node.name}",
                        "response": extract_source(node),
                    }
        except:
            pass
```

### Add Code Quality Filtering

```python
@asset
def high_quality_code(language_filtered_code: Dataset) -> Dataset:
    """Filter by code quality heuristics."""
    def is_high_quality(example):
        code = example["content"]
        # Skip very short files
        if len(code.split()) < 20:
            return False
        # Skip files with excessive comments
        comment_ratio = code.count("#") / max(len(code.split()), 1)
        if comment_ratio > 0.5:
            return False
        return True
    
    return language_filtered_code.filter(is_high_quality)
```

### Mix with Other Code Datasets

```python
@asset
def combined_code_corpus(
    instruction_examples: Dataset,
    other_code_dataset: Dataset,
) -> Dataset:
    """Combine with commercial or proprietary code datasets."""
    from datasets import concatenate_datasets
    return concatenate_datasets([instruction_examples, other_code_dataset])
```

## Use Cases

### Code Generation Model Training
- Fine-tune models like CodeLlama, StarCoder on specific languages
- Instruction-tuning for code task-specific models

### Code Search & Embedding
- Use instruction-response pairs to train dual encoders for code-to-text retrieval

### Automated Code Completion
- Build language-specific code predictors

### Code LLM Evaluation
- Use filtered corpus to evaluate model performance on realistic code distributions

## Integration Ideas

### Combine with Test Data
```python
@asset
def code_with_tests(
    instruction_examples: Dataset,
    test_dataset: Dataset,
) -> Dataset:
    """Pair code with corresponding unit tests."""
    # Match code files to test files
    # Create instruction: "Fix failing test", response: "corrected code"
```

### Dynamic Language Routing (Dagster Sensors)
```python
@sensor(asset_selection=AssetSelection.groups("code_instruction_curation"))
def update_on_dataset_change(context):
    """Auto-refresh when the-stack-dedup updates on Hub."""
    # Trigger full pipeline on new dataset revision
```

## See Also

- [BigCode the-stack-dedup on Hub](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [StarCoder Paper](https://arxiv.org/abs/2305.06161)
- [Related Examples](../):
  - `distributed_token_sharding/` — Token-level processing for code
  - `golden_pipeline/` — Multi-stage cleaning patterns
  - `dynamic_bucket_partitioning/` — Language-based partitioning (natural language)
