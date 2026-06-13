from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
)
from dagster_hf_datasets.io_manager import HFParquetIOManager

from vision_language_scale_pipeline.assets import (
    caption_quality_filtered,
    dedup_quality_report,
    deduplicated_by_caption_hash,
    language_identified_captions,
    raw_laion_coco,
)

defs = Definitions(
    assets=[
        raw_laion_coco,
        caption_quality_filtered,
        deduplicated_by_caption_hash,
        language_identified_captions,
        dedup_quality_report,
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
