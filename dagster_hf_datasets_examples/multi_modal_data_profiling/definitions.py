from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from assets import (
    flickr30k_raw,
    image_stats,
    caption_stats,
    sample_gallery,
    dataset_health_report,
)


defs = Definitions(
    assets=[
        flickr30k_raw,
        image_stats,
        caption_stats,
        sample_gallery,
        dataset_health_report,
    ],
    resources={
        "huggingface": HuggingFaceResource(
            cache_dir=".hf_cache",
            offline=False,
            # token="...",  # nlphuji/flickr30k may require Hub login
        ),
        "hf_parquet_io_manager": HFParquetIOManager(
            base_dir=".dagster_hf_storage",
        ),
    },
)