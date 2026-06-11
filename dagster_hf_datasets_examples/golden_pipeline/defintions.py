from dagster import Definitions

from dagster_hf_datasets import (
    HuggingFaceResource,
    HFParquetIOManager,
)

from assets import (
    c4_raw,
    c4_cleaned,
    c4_quality_validated,
    quality_report,
    c4_train,
    c4_test,
    c4_train_tokenized,
    c4_test_tokenized,
    dataset_card,
    hub_publication_manifest,
)


defs = Definitions(
    assets=[
        c4_raw,
        c4_cleaned,
        c4_quality_validated,
        quality_report,
        c4_train,
        c4_test,
        c4_train_tokenized,
        c4_test_tokenized,
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