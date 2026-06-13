from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from multi_asset_split_routing.assets import (
    glue_sst2,
    glue_sst2_train_normalized,
    split_lineage_report,
)

defs = Definitions(
    assets=[
        glue_sst2,
        glue_sst2_train_normalized,
        split_lineage_report,
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
