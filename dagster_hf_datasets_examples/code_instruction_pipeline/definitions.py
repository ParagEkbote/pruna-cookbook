from dagster import Definitions

from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from code_instruction_pipeline.assets import (
    raw_code_stack,
    language_filtered_code,
    instruction_examples,
    code_quality_metrics,
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
