# Vision Dataset

Prepare a vision-language dataset by validating image-caption rows, splitting
the result, and generating a small dataset card.

## What this example shows

- Loading an image-caption dataset with `@hf_dataset_asset`
- Validating caption fields before downstream use
- Creating reproducible train / validation splits with a fixed seed
- Passing Hugging Face `Dataset` objects between Dagster assets
- Generating a lightweight dataset card from materialized split metadata

## Dataset

[`google-research-datasets/conceptual_captions`](https://huggingface.co/datasets/google-research-datasets/conceptual_captions) -
image URLs paired with natural-language captions. The dataset is a practical
stand-in for vision-language pretraining and retrieval workflows because each
row carries text that can be validated before image fetching or embedding.
The example uses the explicit `unlabeled` config to avoid relying on the Hub
default when multiple subsets are available.

| Asset | Description |
|-------|-------------|
| `conceptual_captions` | Loads the train split from the Hub |
| `validated_pairs` | Keeps rows with a non-empty `caption` |
| `cc_train` | 90% train split |
| `cc_validation` | 10% validation split |
| `dataset_card` | Markdown summary of the generated splits |

## Asset graph

```
conceptual_captions
        |
        v
validated_pairs
     /      \
    v        v
cc_train  cc_validation
     \      /
      v    v
   dataset_card
```

## Validation rule

```python
validated = conceptual_captions.filter(
    lambda ex: (
        ex.get("caption") is not None
        and len(ex["caption"].strip()) > 0
    )
)
```

This keeps the example focused on metadata and caption quality. Production
pipelines often add image URL checks, fetch validation, MIME-type checks, and
deduplication before training.

## Split behavior

Both split assets call:

```python
validated_pairs.train_test_split(
    test_size=0.1,
    seed=42,
)
```

The fixed seed makes the split reproducible across runs as long as the upstream
dataset fingerprint is unchanged.

## Metadata visible in the Dagster UI

| Asset | Key | Description |
|-------|-----|-------------|
| `conceptual_captions` | `rows` | Raw row count |
| `conceptual_captions` | `columns` | Dataset column names |
| `conceptual_captions` | `config` | Source config (`unlabeled`) |
| `conceptual_captions` | `fingerprint` | Hugging Face dataset fingerprint |
| `validated_pairs` | `validated_rows` | Rows with non-empty captions |
| `dataset_card` | `train_rows` | Final train split size |
| `dataset_card` | `validation_rows` | Final validation split size |

## Storage layout

```
.dagster_hf_storage/
├── conceptual_captions/
├── validated_pairs/
├── cc_train/
└── cc_validation/
```

`dataset_card` returns markdown text and metadata. It is not written by the
Hugging Face IO manager.

## How to run

```bash
pip install dagster dagster-hf-datasets datasets
dagster dev -f definitions.py
```

Materialize in order: `conceptual_captions` -> `validated_pairs`, then
`cc_train` and `cc_validation`, and finally `dataset_card`.
