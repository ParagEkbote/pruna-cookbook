import statistics
from collections import Counter

from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset

# ── Language-Specific Code Corpus Filtering & Analysis ──────────────────────


@hf_dataset_asset(
    path="bigcode/the-stack-dedup",
    split="train",
    group_name="code_instruction_curation",
    io_manager_key="hf_parquet_io_manager",
)
def raw_code_stack(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """Ingest BigCode the-stack-dedup dataset.

    The Stack Dedup is a deduplicated collection of 3.1B files across 358
    programming languages. It's used for training large code LLMs like
    StarCoder, CodeLlama, and similar models.

    Demonstrates:
    - Accessing large code corpora
    - Language distribution analysis
    - Sampling for development

    Sample size: 10K files for development (can adjust)
    """
    sample_size = min(10000, len(dataset))
    sampled = dataset.select(range(sample_size))

    # Analyze language distribution in sample
    def get_lang(ex):
        return ex.get("lang", ex.get("language", "unknown"))

    # Sample a subset for language analysis
    lang_sample_indices = list(range(0, min(1000, len(sampled))))
    lang_counts = Counter()

    for i in lang_sample_indices:
        lang = get_lang(sampled[i])
        lang_counts[lang] += 1

    top_langs = dict(lang_counts.most_common(10))

    context.log.info(
        "Loaded the-stack-dedup sample: %s files, top 10 languages: %s",
        len(sampled),
        top_langs,
    )

    return MaterializeResult(
        value=sampled,
        metadata={
            "rows": len(sampled),
            "total_dataset_size": len(dataset),
            "languages_in_sample": len(lang_counts),
            "top_languages": top_langs,
            "source_dataset": "bigcode/the-stack-dedup",
        },
    )


@asset(
    group_name="code_instruction_curation",
    io_manager_key="hf_parquet_io_manager",
)
def language_filtered_code(
    context: AssetExecutionContext,
    raw_code_stack: Dataset,
) -> MaterializeResult:
    """Filter code to high-value languages for LLM instruction-tuning.

    Target languages: Python, JavaScript, Go, Java, Rust, TypeScript, C++
    These are widely used in industry and have good quality code samples.
    """
    target_languages = {
        "Python",
        "JavaScript",
        "Go",
        "Java",
        "Rust",
        "TypeScript",
        "C++",
    }

    def is_target_language(example):
        lang = example.get("lang", example.get("language"))
        return lang in target_languages

    filtered = raw_code_stack.filter(is_target_language)

    retention_pct = round((len(filtered) / len(raw_code_stack)) * 100, 2)

    context.log.info(
        "Filtered to target languages: %s / %s files (%.1f%% retained)",
        len(filtered),
        len(raw_code_stack),
        retention_pct,
    )

    context.add_output_metadata(
        {
            "input_rows": len(raw_code_stack),
            "output_rows": len(filtered),
            "retention_pct": retention_pct,
            "target_languages": list(target_languages),
        }
    )

    return MaterializeResult(value=filtered, metadata={"rows": len(filtered)})


@asset(
    group_name="code_instruction_curation",
    io_manager_key="hf_parquet_io_manager",
)
def instruction_examples(
    context: AssetExecutionContext,
    language_filtered_code: Dataset,
) -> MaterializeResult:
    """Extract code snippets as instruction-response pairs.

    Treats code files as:
    - Instruction: "Implement a function called {function_name}"
    - Response: The actual code content

    Suitable for fine-tuning code generation models.
    """
    records = []

    for i, example in enumerate(language_filtered_code):
        code = example.get("content", "")
        lang = example.get("lang", example.get("language", "unknown"))

        if not code or len(code.split()) < 10:
            continue

        # Simple extraction: treat entire file as response
        # In production, parse ASTs to extract function/class definitions
        records.append(
            {
                "instruction": f"Write {lang} code that solves the following problem.",
                "response": code,
                "language": lang,
                "code_length": len(code),
                "token_count": len(code.split()),
            }
        )

        if i % 1000 == 0:
            context.log.info("Processed %s / %s files", i, len(language_filtered_code))

    if not records:
        context.log.warning("No instruction examples extracted!")
        records = []

    # Create dataset from records
    from datasets import Dataset as HFDataset

    instruction_dataset = HFDataset.from_list(records)

    token_counts = [r["token_count"] for r in records] if records else [0]

    context.log.info(
        "Extracted %s instruction examples (avg %.0f tokens)",
        len(records),
        statistics.mean(token_counts) if token_counts else 0,
    )

    context.add_output_metadata(
        {
            "instruction_count": len(records),
            "avg_tokens": round(statistics.mean(token_counts), 1) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
        }
    )

    return MaterializeResult(
        value=instruction_dataset,
        metadata={
            "rows": len(records),
        },
    )


@asset(group_name="code_instruction_curation")
def code_quality_metrics(
    context: AssetExecutionContext,
    raw_code_stack: Dataset,
    language_filtered_code: Dataset,
    instruction_examples: Dataset,
) -> MaterializeResult:
    """Compute dataset quality and diversity metrics.

    Provides visibility into:
    - Language distribution across pipeline stages
    - Code length distribution
    - Data retention percentages
    - Quality indicators
    """
    # Language distribution pre-filter
    lang_counts_raw = Counter()
    for i in range(min(1000, len(raw_code_stack))):
        lang = raw_code_stack[i].get("lang", raw_code_stack[i].get("language", "unknown"))
        lang_counts_raw[lang] += 1

    # Language distribution post-filter
    lang_counts_filtered = Counter()
    for i in range(min(len(language_filtered_code), 1000)):
        lang = language_filtered_code[i].get(
            "lang",
            language_filtered_code[i].get("language", "unknown"),
        )
        lang_counts_filtered[lang] += 1

    report = {
        "raw_files": len(raw_code_stack),
        "after_language_filter": len(language_filtered_code),
        "language_filter_retention_pct": round(
            (len(language_filtered_code) / len(raw_code_stack)) * 100, 2
        ),
        "instruction_examples": len(instruction_examples),
        "instruction_extraction_rate": round(
            (len(instruction_examples) / len(language_filtered_code)) * 100, 2
        ),
        "top_5_languages_post_filter": dict(lang_counts_filtered.most_common(5)),
        "quality_score": round(
            (len(instruction_examples) / len(raw_code_stack)) * 100, 1
        ),
    }

    context.log.info("Code quality metrics: %s", report)

    return MaterializeResult(
        value=report,
        metadata=report,
    )
