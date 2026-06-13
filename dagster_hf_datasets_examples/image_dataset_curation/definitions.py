from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource
from dagster_hf_datasets.io_manager import HFParquetIOManager

from image_dataset_curation.assets import (
    aspect_ratio_validated,
    check_aspect_bounds,
    check_curation_retention,
    corrupt_removed,
    curated_export,
    curation_report,
    deduplicated_images,
    resolution_filtered,
    tiny_imagenet_raw,
)

defs = Definitions(
    assets=[
        tiny_imagenet_raw,
        resolution_filtered,
        corrupt_removed,
        deduplicated_images,
        aspect_ratio_validated,
        curated_export,
        curation_report,
    ],
    asset_checks=[
        check_curation_retention,
        check_aspect_bounds,
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
