from dagster import AssetExecutionContext, MaterializeResult
from dagster_hf_datasets import hf_dataset_asset
from datasets import Dataset

MIN_DURATION = 1.0
MAX_DURATION = 20.0


@hf_dataset_asset(
    path="mozilla-foundation/common_voice_11_0",
    config="en",
    split="train",
    group_name="audio_dataset_curation",
    io_manager_key="hf_parquet_io_manager",
)
def curated_audio_dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
) -> MaterializeResult:
    """
    Curate Common Voice English speech clips.

    Demonstrates:
    - Duration filtering
    - Audio validation
    - Language-tag validation
    """

    initial_rows = len(dataset)

    def is_valid(example):
        audio = example.get("audio")

        if audio is None:
            return False

        if "array" not in audio:
            return False

        if audio["array"] is None:
            return False

        duration = audio.get("duration")

        if duration is None:
            return False

        if duration < MIN_DURATION:
            return False

        if duration > MAX_DURATION:
            return False

        locale = example.get("locale")

        if locale and not locale.startswith("en"):
            return False

        return True

    curated_dataset = dataset.filter(is_valid)

    removed_rows = initial_rows - len(curated_dataset)

    context.log.info(
        "Filtered %s invalid audio samples",
        removed_rows,
    )

    return MaterializeResult(
        value=curated_dataset,
        metadata={
            "original_rows": initial_rows,
            "curated_rows": len(curated_dataset),
            "removed_rows": removed_rows,
            "dataset": "mozilla-foundation/common_voice_11_0",
            "split": "train",
            "fingerprint": curated_dataset._fingerprint,
        },
    )
