from dagster import Definitions
from dagster_hf_datasets import (
    HuggingFaceResource,
)
from dagster_hf_datasets.io_manager import HFParquetIOManager

from golden_pipeline.assets import (
    dataset_card,
    fineweb_cleaned,
    fineweb_quality_validated,
    fineweb_raw,
    fineweb_test,
    fineweb_test_tokenized,
    fineweb_train,
    fineweb_train_tokenized,
    hub_publication_manifest,
    quality_report,
)

defs = Definitions(
    assets=[
        fineweb_raw,
        fineweb_cleaned,
        fineweb_quality_validated,
        quality_report,
        fineweb_train,
        fineweb_test,
        fineweb_train_tokenized,
        fineweb_test_tokenized,
        dataset_card,
        hub_publication_manifest,
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
