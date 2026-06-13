from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from multi_modal_data_profiling.assets import (
    caption_stats,
    dataset_health_report,
    flickr30k_raw,
    image_stats,
    llava_instruct_raw,
    llava_instruction_stats,
    llava_quality_profile,
    sample_gallery,
)

defs = Definitions(
    assets=[
        flickr30k_raw,
        image_stats,
        caption_stats,
        sample_gallery,
        dataset_health_report,
        llava_instruct_raw,
        llava_instruction_stats,
        llava_quality_profile,
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
