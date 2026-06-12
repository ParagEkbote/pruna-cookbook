from datasets import load_dataset

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    {
        "csv_path": "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_hqq.csv",
        "repo_id": "AINovice2005/raw_benchmark_results_hqq",
        "parquet_path": "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_hqq.parquet",
    },
    {
        "csv_path": "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_pruna.csv",
        "repo_id": "AINovice2005/raw_benchmark_results_pruna",
        "parquet_path": "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_pruna.parquet",
    },
]

PRIVATE = False

# ============================================================================
# Process and Upload
# ============================================================================

for config in DATASETS:
    print(f"\nProcessing: {config['repo_id']}")

    ds = load_dataset(
        "csv",
        data_files=config["csv_path"],
        split="train",
    )

    print(ds)

    # Save local parquet copy
    ds.to_parquet(config["parquet_path"])
    print(f"✓ Saved parquet: {config['parquet_path']}")

    # Push to Hub
    ds.push_to_hub(
        repo_id=config["repo_id"],
        private=PRIVATE,
    )

    print(f"✓ Uploaded: https://huggingface.co/datasets/{config['repo_id']}")

print("\nAll uploads complete.")