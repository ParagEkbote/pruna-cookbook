from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
)
from dagster_hf_datasets.io_manager import HFParquetIOManager

from code_instruction_pipeline.assets import (
    code_quality_metrics,
    instruction_examples,
    language_filtered_code,
    raw_code_stack,
)

defs = Definitions(
    assets=[
        raw_code_stack,
        language_filtered_code,
        instruction_examples,
        code_quality_metrics,
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
