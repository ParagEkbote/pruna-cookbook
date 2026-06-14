# dagster-hf-datasets Examples

A curated collection of 17 end-to-end examples demonstrating how to build reproducible, observable, and production-ready ML data pipelines with `dagster-hf-datasets`

Each example is self-contained, runnable with `dagster dev -m  <example_folder>.definitions.py`, and covers a distinct concept or workflow tier — from basic Hub ingestion to flagship end-to-end pipelines.

---

## Prerequisites

```bash
pip install dagster dagster-hf-datasets
```

Some examples require additional dependencies (noted in their README). Set your Hugging Face token for gated datasets:

```bash
export HF_TOKEN=hf_...
```

All examples share the same `Definitions` boilerplate. The `HuggingFaceResource` handles Hub loading; `HFParquetIOManager` handles persistence:

```python
from dagster import Definitions
from dagster_hf_datasets import HuggingFaceResource, HFParquetIOManager

defs = Definitions(
    assets=[...],
    resources={
        "huggingface": HuggingFaceResource(cache_dir=".hf_cache"),
        "hf_parquet_io_manager": HFParquetIOManager(base_dir=".dagster_hf_storage"),
    },
)
```

---

## Running an Example

```bash
cd dagster_hf_datasets_examples

dagster dev -m  <example_folder>.definitions.py
```

Every `definitions.py` is a standalone entrypoint.

---

## Repository Structure

```
dagster-hf-datasets-examples/
├── README.md                      ← you are here
├── pyproject.toml
│
├── basic_hub_ingestion/
│   ├── assets.py                  ← @hf_dataset_asset / @hf_multi_asset definitions
│   ├── definitions.py             ← Definitions() entrypoint for dagster dev
│   ├── README.md                  ← goal, dataset, API notes, expected output
│   └── __init__.py
│
└── <example_name>/                ← same layout for all 17 examples
    ├── assets.py
    ├── definitions.py
    ├── README.md
    └── __init__.py
```

---

## Example Catalog

### 🏗️ Foundation

Core patterns every `dagster-hf-datasets` user encounters first.

| Folder | Description | Dataset | Key API |
|--------|-------------|---------|---------|
| [`basic_hub_ingestion`](./basic_hub_ingestion/) | Load a Hub dataset and materialize it as a Dagster asset with lineage metadata | `stanfordnlp/imdb` | `@hf_dataset_asset` |
| [`multi_asset_split_routing`](./multi_asset_split_routing/) | Materialize train/validation/test splits as independently tracked assets | `nyu-mll/glue` (sst2) | `@hf_multi_asset` |
| [`sanitization_observability`](./sanitization_observability/) | Filter, deduplicate, and expose quality metrics via asset checks | `HuggingFaceFW/fineweb-edu` | `@asset_check`, `Dataset.filter()` |
| [`dataset_card_publishing`](./dataset_card_publishing/) | Generate and publish a Hub README from pipeline metadata | `rajpurkar/squad` | `HFDatasetPublisher` |

---

### 🔄 Data Management & Automation

Patterns for scalability, automation, and memory-safe processing.

| Folder | Description | Dataset | Key API |
|--------|-------------|---------|---------|
| [`event_driven_benchmark_refresh`](./event_driven_benchmark_refresh/) | Re-materialize evaluation datasets automatically when an upstream Hub revision changes | `openai/openai_humaneval` | Dagster sensors |
| [`streaming_oom_datasets`](./streaming_oom_datasets/) | Process datasets larger than available memory using `IterableDataset` | `allenai/dolma` | `IterableDataset`, chunked Parquet writes |
| [`dynamic_bucket_partitioning`](./dynamic_bucket_partitioning/) | Partition datasets by language, domain, or task with Dagster's partition system | `Helsinki-NLP/opus_books` | `StaticPartitionsDefinition`, `DynamicPartitionsDefinition` |
| [`distributed_token_sharding`](./distributed_token_sharding/) | Pre-tokenize large datasets into training-ready shards | `HuggingFaceFW/fineweb` | Batched `Dataset.map()`, `.npy` shard caching |

---

### 🖼️ Multimodal Workflows

Image, audio, and vision-language pipelines.

| Folder | Description | Dataset | Key API |
|--------|-------------|---------|---------|
| [`multi_modal_data_profiling`](./multi_modal_data_profiling/) | Profile an image-text dataset: resolution stats, caption analysis, thumbnail gallery, health score | `nlphuji/flickr30k` | PIL, `Dataset.from_list()` |
| [`image_dataset_curation`](./image_dataset_curation/) | Build a production-quality vision dataset: corrupt detection, dedup, aspect-ratio validation, multi-format export | `zh-plus/tiny-imagenet` | `@multi_asset`, WebDataset/Arrow/Parquet packaging |
| [`audio_dataset_curation`](./audio_dataset_curation/) | Construct a clean speech corpus: duration filtering, SNR gating, language validation | `mozilla-foundation/common_voice_11_0` | `soundfile`, `librosa` |
| [`vision_dataset`](./vision_dataset/) | Package image datasets into training-optimized formats with manifests | `zh-plus/tiny-imagenet` | WebDataset `.tar`, Arrow, Parquet |
| [`synthethic_multimodal_data`](./synthethic_multimodal_data/) | Caption pre-generated synthetic images, score caption↔prompt alignment, filter by quality | `poloclub/diffusiondb` | BLIP-base (CPU), MiniLM similarity |
| [`vision_language_scale_pipeline`](./vision_language_scale_pipeline/) | Build CLIP/LLaVA-style image-caption datasets at scale | `google-research-datasets/conceptual_captions` | Image-caption pairing, schema validation |

---

### 🤖 LLM & Alignment Workflows

Text, instruction, and preference-alignment pipelines.

| Folder | Description | Dataset | Key API |
|--------|-------------|---------|---------|
| [`code_instruction_pipeline`](./code_instruction_pipeline/) | Prepare code and instruction-following datasets for model training | `bigcode/the-stack` | Deduplication, language filtering |
| [`preference_alignment_data`](./preference_alignment_data/) | Build RLHF/DPO-ready datasets from raw preference records | `Anthropic/hh-rlhf` | Prompt/chosen/rejected schema, pair validation |
| [`golden_pipeline`](./golden_pipeline/) | Complete end-to-end pipeline: ingest → clean → validate → split → tokenize → publish | `allenai/c4` | Full asset graph, `HFDatasetPublisher` |

---

## Learning Path

### Beginner — start here if you're new to dagster-hf-datasets

1. [`basic_hub_ingestion`](./basic_hub_ingestion/) — understand `@hf_dataset_asset` and `MaterializeResult`
2. [`multi_asset_split_routing`](./multi_asset_split_routing/) — see how `@hf_multi_asset` auto-resolves splits
3. [`sanitization_observability`](./sanitization_observability/) — chain `@hf_dataset_asset` with downstream `@asset` nodes
4. [`dataset_card_publishing`](./dataset_card_publishing/) — publish lineage-aware Hub documentation

### Intermediate — build operational pipelines

5. [`event_driven_benchmark_refresh`](./event_driven_benchmark_refresh/) — add sensor-based automation
6. [`streaming_oom_datasets`](./streaming_oom_datasets/) — handle datasets that exceed memory
7. [`dynamic_bucket_partitioning`](./dynamic_bucket_partitioning/) — partition by language/domain/task
8. [`distributed_token_sharding`](./distributed_token_sharding/) — generate training-ready tokenized shards

### Advanced — multimodal and specialized workflows

9. [`multi_modal_data_profiling`](./multi_modal_data_profiling/)
10. [`image_dataset_curation`](./image_dataset_curation/)
11. [`audio_dataset_curation`](./audio_dataset_curation/)
12. [`vision_dataset`](./vision_dataset/)
13. [`synthethic_multimodal_data`](./synthethic_multimodal_data/)
14. [`vision_language_scale_pipeline`](./vision_language_scale_pipeline/)
15. [`code_instruction_pipeline`](./code_instruction_pipeline/)
16. [`preference_alignment_data`](./preference_alignment_data/)
17. [`golden_pipeline`](./golden_pipeline/) — run this last to see everything together

---


## Common Gotchas

**The decorated function body is not responsible for loading the dataset.**
`HuggingFaceResource` performs the load; the function receives the injected `dataset` parameter. Your function body inspects, transforms, and returns `MaterializeResult`.

**`HFParquetIOManager` does not persist `IterableDataset`.**
Streaming datasets must be written to Parquet manually via `pyarrow`. See `streaming_oom_datasets` for the pattern.

**`HFDatasetPublisher` is in a private module.**
Import from `dagster_hf_datasets._export._publisher`. Pin your `dagster-hf-datasets` version if you use this.

**Splits from `@hf_multi_asset` are named `{asset_name}_{split}`.**
A `glue_sst2` multi-asset produces `glue_sst2_train`, `glue_sst2_validation`, `glue_sst2_test` as downstream dependency names.

**Split before tokenizing.**
Always train-test split before tokenization to avoid data leakage. The `golden_pipeline` demonstrates the correct order.

---

## Resources

- [dagster-hf-datasets on GitHub](https://github.com/dagster-io/community-integrations/tree/main/libraries/dagster-hf-datasets)
- [API Reference](https://github.com/dagster-io/community-integrations/blob/main/libraries/dagster-hf-datasets/docs/API.md)
- [Usage Guide](https://github.com/dagster-io/community-integrations/blob/main/libraries/dagster-hf-datasets/docs/Usage.md)
- [Dagster Docs: HF Datasets Integration](https://docs.dagster.io/integrations/libraries/hf-datasets)
- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets)