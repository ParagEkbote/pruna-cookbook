from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from assets import (
    raw_fineweb_edu,
    filtered_fineweb_edu,
    deduplicated_fineweb_edu,
    cleaning_quality_report,
    check_no_null_text,
    check_retention_rate,
)


defs = Definitions(
    assets=[
        raw_fineweb_edu,
        filtered_fineweb_edu,
        deduplicated_fineweb_edu,
        cleaning_quality_report,
    ],
    asset_checks=[
        check_no_null_text,
        check_retention_rate,
    ],
    resources={
        "huggingface": HuggingFaceResource(
            cache_dir=".hf_cache",
            offline=False,
            # token="...",  # or set HF_TOKEN env var
        ),
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)