from itertools import islice

from dagster import AssetExecutionContext, MaterializeResult, asset
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset

STREAM_SAMPLE_SIZE = 1000


@hf_dataset_asset(
    path="allenai/dolma",
    split="train",
    streaming=True,
    group_name="streaming_oom_datasets",
    io_manager_key="hf_parquet_io_manager",
)
def dolma_stream(
    context: AssetExecutionContext,
    dataset,
) -> MaterializeResult:
    """Read a bounded sample from a streamed text corpus.

    Streaming datasets do not support len() or random access. This asset keeps
    the materialized output intentionally small so local runs do not try to
    download or persist the full source corpus.
    """
    rows = list(islice(dataset, STREAM_SAMPLE_SIZE))
    sample = Dataset.from_list(rows)

    context.log.info("Streamed %s rows from allenai/dolma", len(sample))

    return MaterializeResult(
        value=sample,
        metadata={
            "sample_rows": len(sample),
            "processing_limit": STREAM_SAMPLE_SIZE,
            "source_dataset": "allenai/dolma",
            "streaming": True,
        },
    )


@asset(
    group_name="streaming_oom_datasets",
)
def streaming_report(
    context: AssetExecutionContext,
    dolma_stream: Dataset,
) -> MaterializeResult:
    text_lengths = [
        len(example.get("text", ""))
        for example in dolma_stream
    ]

    report = {
        "sample_rows": len(dolma_stream),
        "avg_text_chars": round(sum(text_lengths) / len(text_lengths), 1)
        if text_lengths else 0.0,
        "max_text_chars": max(text_lengths) if text_lengths else 0,
        "streaming": True,
    }

    context.log.info("Streaming sample report: %s", report)

    return MaterializeResult(
        value=report,
        metadata=report,
    )
