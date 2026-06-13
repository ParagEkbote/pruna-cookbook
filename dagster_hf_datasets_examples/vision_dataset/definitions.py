from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
)
from dagster_hf_datasets.io_manager import HFParquetIOManager

from vision_dataset.assets import (
    cc_train,
    cc_validation,
    conceptual_captions,
    dataset_card,
    validated_pairs,
)

defs = Definitions(
    assets=[
        conceptual_captions,
        validated_pairs,
        cc_train,
        cc_validation,
        dataset_card,
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
