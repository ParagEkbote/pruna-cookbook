from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from dataset_card_publishing.assets import (
    publish_squad_dataset,
    squad_enriched,
    squad_train,
)

defs = Definitions(
    assets=[
        squad_train,
        squad_enriched,
        publish_squad_dataset,
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
