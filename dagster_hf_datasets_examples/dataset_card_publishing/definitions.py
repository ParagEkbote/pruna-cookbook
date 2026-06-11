from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from assets import (
    squad_train,
    squad_enriched,
    squad_dataset_card,
)


defs = Definitions(
    assets=[
        squad_train,
        squad_enriched,
        squad_dataset_card,
    ],
    resources={
        "huggingface": HuggingFaceResource(
            cache_dir=".hf_cache",
            offline=False,
            # token="...",  # or set HF_TOKEN env var — required for publishing
        ),
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)