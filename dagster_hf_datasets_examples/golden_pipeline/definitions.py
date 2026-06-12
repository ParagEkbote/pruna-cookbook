from dagster import Definitions

from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from golden_pipeline.assets import (
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