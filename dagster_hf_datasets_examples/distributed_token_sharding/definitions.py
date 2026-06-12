from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from distributed_token_sharding.assets import (
    fineweb_dataset,
    tokenized_fineweb,
)


defs = Definitions(
    assets=[
        fineweb_dataset,
        tokenized_fineweb,
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