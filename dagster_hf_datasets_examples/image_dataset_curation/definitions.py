from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from image_dataset_curation.assets import (
    humaneval_benchmark,
    humaneval_formatted,
    benchmark_refresh_report,
    humaneval_revision_sensor,
)


defs = Definitions(
    assets=[
        humaneval_benchmark,
        humaneval_formatted,
        benchmark_refresh_report,
    ],
    sensors=[
        humaneval_revision_sensor,
    ],
    resources={
        "huggingface": HuggingFaceResource(
            cache_dir=".hf_cache",
            offline=False,
        ),
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)