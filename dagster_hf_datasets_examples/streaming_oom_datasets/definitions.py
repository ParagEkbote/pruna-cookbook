from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
)
from dagster_hf_datasets.io_manager import HFParquetIOManager

from streaming_oom_datasets.assets import (
    dolma_stream,
    streaming_report,
)


defs = Definitions(
    assets=[
        dolma_stream,
        streaming_report,
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