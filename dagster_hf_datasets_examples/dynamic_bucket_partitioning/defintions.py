from dagster import Definitions

from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from assets import (
    language_partitions,
    opus_books_raw,
    partition_report,
)


defs = Definitions(
    assets=[
        opus_books_raw,
        partition_report,
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