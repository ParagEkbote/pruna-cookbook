from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from vision_dataset.assets import (
    conceptual_captions,
    validated_pairs,
    cc_train,
    cc_validation,
    dataset_card,
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