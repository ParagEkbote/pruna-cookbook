from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from synthethic_multimodal_data.assets import (
    caption_alignment_scores,
    check_mean_alignment,
    check_no_empty_captions,
    diffusiondb_sample,
    nsfw_filtered,
    synthetic_captions,
    synthetic_dataset_final,
    synthetic_generation_report,
)

defs = Definitions(
    assets=[
        diffusiondb_sample,
        nsfw_filtered,
        synthetic_captions,
        caption_alignment_scores,
        synthetic_dataset_final,
        synthetic_generation_report,
    ],
    asset_checks=[
        check_no_empty_captions,
        check_mean_alignment,
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
