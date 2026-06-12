1. Always Return MaterializeResult for Example Assets
Problem

Asset executes successfully but UI shows:

Never materialized

or metadata appears inconsistently.

Bad
@asset
def report():
    return {"rows": 100}
Good
return MaterializeResult(
    value=report,
    metadata={
        "rows": 100,
    },
)
Recommendation

For cookbook examples:

Always return MaterializeResult.

even when not strictly required.

2. Asset Dependencies Must Be Type-Compatible
Problem
Input type mismatch

or loading failures.

Bad
@asset
def cleaned_dataset(raw_dataset):
    ...

without clear typing.

Good
@asset
def cleaned_dataset(
    raw_dataset: Dataset,
) -> MaterializeResult:

Use HF Dataset consistently across examples.

3. Use the Same IO Manager Everywhere
Problem

Downstream asset can't load upstream output.

Bad
asset_a -> hf_parquet_io_manager
asset_b -> default_io_manager
Good
io_manager_key="hf_parquet_io_manager"

for all dataset-producing assets.

4. Dataset Objects Must Be Returned
Problem

Downstream asset expects:

Dataset

but receives:

dict
list
DataFrame
Example

Bad:

return dataset.to_pandas()

Good:

return MaterializeResult(
    value=dataset,
)

unless the example explicitly demonstrates conversion.

5. Materialization Metadata Must Be JSON Serializable
Problem
Metadata serialization error

Bad:

metadata={
    "features": dataset.features
}

Good:

metadata={
    "features": str(dataset.features)
}

or

metadata={
    "columns": dataset.column_names
}
6. Dataset Fingerprints Can Change
Problem

Users assume fingerprints are stable forever.

dataset._fingerprint

changes when:

filtering
mapping
shuffling

Document this.

7. Streaming Datasets Behave Differently
Problem

Code works locally.

Fails for:

streaming=True

because:

len(dataset)

is unsupported.

Example

Avoid streaming in examples unless teaching streaming.

8. Hugging Face Cache Confusion
Problem

Users think dataset is redownloaded.

Actually loaded from:

~/.cache/huggingface

or

cache_dir=".hf_cache"

Document:

Delete .hf_cache to force a re-download.
9. Asset Keys Are Stable API
Problem

Rename asset:

humaneval_benchmark

to:

humaneval_raw

and lose lineage/history.

Document:

Asset names become Dagster asset keys.
Changing them creates a new asset.
10. Sensors Need State
Problem

Sensor fires every tick.

Bad:

yield RunRequest(...)

Good:

context.cursor

or:

context.update_cursor(...)
11. Hub Access Can Fail
Problem

Examples break because:

429
503
network timeout
Recommendation

Wrap Hub access:

try:
    ...
except Exception:
    ...

especially for sensors.

12. Private Datasets Need Tokens
Problem

Example works for maintainers but not users.

Document:

export HF_TOKEN=...

or:

HuggingFaceResource(
    token="..."
)
13. Large Dataset Examples Can Surprise Users
Problem

User runs:

load_dataset(...)

and downloads several GB.

Add expected sizes to READMEs.

Example:

Dataset Size
------------
IMDb train: ~80 MB
14. Relative Paths Depend on Working Directory
Problem

Running from:

repo root

works.

Running from:

example folder

fails.

Bad:

base_dir=".dagster_hf_storage"

without explanation.

Document where outputs appear.

15. Dagster Version Drift
Problem

Examples break after Dagster upgrade.

You already found one:

return dict

behaved differently in UI.

Recommendation

Pin:

dagster>=1.13,<1.14

for examples.

16. Dataset.map() Must Return Valid Records
Problem

Users accidentally return:

None

or inconsistent schema.

Bad:

lambda x: None

Good:

lambda x: {
    ...
}
17. Audio Datasets Are Special
Problem

Users assume:

audio["duration"]

always exists.

Many HF audio datasets only provide:

audio["array"]
audio["sampling_rate"]

Duration often must be computed.

18. Materializing One Asset Doesn't Always Materialize Downstream Assets

Many newcomers expect:

humaneval_benchmark
    ↓
humaneval_formatted
    ↓
report

to all run automatically.

Dagster only runs what was selected.

Document:

Materialize All

vs

Materialize Selected
19. Definitions Import Errors

Most common issue:

from assets import ...

fails depending on cwd.

Prefer:

from event_driven_benchmark_refresh.assets import ...

once examples become packages.