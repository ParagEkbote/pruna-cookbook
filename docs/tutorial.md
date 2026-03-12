# Controlled Synthetic Dataset Generation with FLUX, Pruna, DSPy, and SmolVLM

## Overview

This tutorial builds a **closed-loop synthetic dataset distillation system** that generates 12–24 high-quality, curated images through:

1. **Model Optimization** (Pruna SmashConfig) → Fast inference
2. **Prompt Optimization** (DSPy) → Better prompts
3. **Quality Grading** (SmolVLM via VLM evaluator) → Filtered outputs
4. **Top-K Selection** → Only best images kept

**Key insight**: We use the VLM purely as an **evaluator**, not a generator. This ensures clean separation of concerns and reproducible results.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Part A: Optimize FLUX.2-klein-4B with Pruna](#part-a-optimize-flux2-klein-4b-with-pruna)
3. [Part B: Configure VLM Grader (SmolVLM + DSPy)](#part-b-configure-vlm-grader-smolvlm--dspy)
4. [Part C: Define Grading Schema](#part-c-define-grading-schema)
5. [Part D: Dataset Generation Loop](#part-d-dataset-generation-loop)
6. [Part E: Export and Analyze](#part-e-export-and-analyze)
7. [Optional: Fine-Tuning Setup](#optional-fine-tuning-setup)

---

## 1. Installation & Setup

### Prerequisites

```bash
pip install pruna
pip install dspy-ai
pip install transformers
pip install pillow
pip install datasets
pip install torch torchvision torchaudio
```

### Imports

```python
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from PIL import Image
import numpy as np

# Pruna
from pruna import smash, SmashConfig, PrunaModel

# DSPy
import dspy
from dspy import settings

# Datasets
import datasets

# Utilities
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Device & Cache Setup

```python
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Create output directory
OUTPUT_DIR = Path("./flux_dataset_output")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

METADATA_PATH = OUTPUT_DIR / "metadata.json"
```

---

## Part A: Optimize FLUX.2-klein-4B with Pruna

### A1: Load Base Model

```python
def load_base_model():
    """
    Load FLUX.2-klein-4B model.
    
    Note: This requires HF token if model is gated.
    Set HF_TOKEN env var or use huggingface-cli login
    """
    logger.info("Loading FLUX.2-klein-4B base model...")
    
    try:
        # Option 1: From HuggingFace Hub (recommended)
        from diffusers import FluxPipeline
        
        model = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Ensure you have HF token and accept model license")
        raise
    
    logger.info("✓ Base model loaded")
    return model

# Load it
base_model = load_base_model()
```

### A2: Define Pruna SmashConfig

```python
def create_smash_config(batch_size: int = 1) -> SmashConfig:
    """
    Create optimized SmashConfig for FLUX.2-klein-4B.
    
    Strategy:
    - deepcache: Reuse transformer blocks across diffusion steps
    - torch_compile: Compile with inductor backend
    
    Budget: Prioritize speed over aggressive quantization
    (we want quality for curated dataset)
    """
    logger.info("Creating SmashConfig...")
    
    smash_cfg = SmashConfig(
        batch_size=batch_size,
        device=device,
        cache_dir_prefix=str(OUTPUT_DIR / ".cache")
    )
    
    # Add algorithms
    smash_cfg.add("deepcache")
    
    # Configure deepcache
    smash_cfg["deepcache_interval"] = 2  # Balance speed vs quality
    
    # Add torch compile for additional speedup
    smash_cfg.add("torch_compile")
    smash_cfg["torch_compile_backend"] = "inductor"
    smash_cfg["torch_compile_mode"] = "reduce-overhead"
    
    logger.info(f"✓ SmashConfig created with algorithms: {smash_cfg.get_active_algorithms()}")
    return smash_cfg

smash_cfg = create_smash_config()
```

### A3: Apply Pruna Smash

```python
def smash_model(model, smash_cfg: SmashConfig):
    """
    Apply Pruna optimizations to FLUX model.
    
    This reduces:
    - Latency by ~2-3x
    - Memory by ~10-15%
    
    Why? So we can afford multiple candidate generations per subject.
    """
    logger.info("Applying Pruna smash optimizations...")
    
    smashed_model = smash(
        model=model,
        smash_config=smash_cfg
    )
    
    logger.info("✓ Model smashed")
    return smashed_model

# Apply optimization
flux_smashed = smash_model(base_model, smash_cfg)
```

### A4: Save Optimized Model

```python
def save_smashed_model(model, save_path: Path):
    """Save smashed model for reuse"""
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving smashed model to {save_path}...")
    
    model.save_pretrained(str(save_path))
    logger.info("✓ Model saved")

SMASHED_MODEL_PATH = OUTPUT_DIR / "flux_klein_smashed"
save_smashed_model(flux_smashed, SMASHED_MODEL_PATH)
```

---

## Part B: Configure VLM Grader (SmolVLM + DSPy)

### B1: Initialize SmolVLM via DSPy

```python
def configure_dspy_vlm():
    """
    Configure DSPy to use SmolVLM-Instruct as the VLM backend.
    
    SmolVLM is lightweight (~600M params), perfect for repeated evals.
    
    Note: This will download the model on first run (~2GB)
    """
    logger.info("Configuring DSPy with SmolVLM...")
    
    # Define the VLM settings
    lm_config = dspy.HFModel(
        model="HuggingFaceTB/SmolVLM-Instruct",
        hf_type="vision_model",  # Important: tells DSPy this is multimodal
        temperature=0.0,  # Deterministic grading
        top_p=1.0,
        max_tokens=256,
    )
    
    # Configure DSPy globally
    settings.configure(lm=lm_config)
    
    logger.info("✓ DSPy configured with SmolVLM")

configure_dspy_vlm()
```

---

## Part C: Define Grading Schema

### C1: Image Grade Signature

```python
class ImageGrade(dspy.Signature):
    """
    Structured grading signature for VLM evaluator.
    
    Dimensions:
    - prompt_adherence (0-10): How well does image match the prompt?
    - aesthetic_quality (0-10): Composition, colors, clarity
    - text_correctness (0-10): If prompt mentions text, is it rendered correctly?
    - brand_alignment (0-10): Does it feel professional/on-brand?
    """
    
    prompt: str = dspy.InputField(
        desc="The text prompt used to generate the image"
    )
    image: object = dspy.InputField(
        desc="PIL Image object to evaluate"
    )
    
    prompt_adherence: float = dspy.OutputField(
        desc="Score 0-10: Alignment between prompt and generated image"
    )
    aesthetic_quality: float = dspy.OutputField(
        desc="Score 0-10: Visual quality, composition, clarity"
    )
    text_correctness: float = dspy.OutputField(
        desc="Score 0-10: Correctness of text rendering (if any)"
    )
    brand_alignment: float = dspy.OutputField(
        desc="Score 0-10: Professional appearance, brand fitness"
    )
```

### C2: VLM Grader Module

```python
class VLMGrader(dspy.Module):
    """
    Module that uses SmolVLM to grade images.
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ImageGrade)
    
    def forward(self, prompt: str, image: Image.Image) -> Dict[str, float]:
        """
        Grade a single image.
        
        Args:
            prompt: The original generation prompt
            image: PIL Image object
        
        Returns:
            Dictionary with scores for each dimension
        """
        prediction = self.predictor(prompt=prompt, image=image)
        
        return {
            "prompt_adherence": float(prediction.prompt_adherence),
            "aesthetic_quality": float(prediction.aesthetic_quality),
            "text_correctness": float(prediction.text_correctness),
            "brand_alignment": float(prediction.brand_alignment),
        }

# Instantiate grader
vlm_grader = VLMGrader()
```

### C3: Composite Scoring Metric

```python
class ScoringWeights:
    """Weights for composite score calculation"""
    PROMPT_ADHERENCE = 0.40
    AESTHETIC_QUALITY = 0.30
    TEXT_CORRECTNESS = 0.20
    BRAND_ALIGNMENT = 0.10

def compute_composite_score(scores: Dict[str, float]) -> float:
    """
    Compute weighted composite score.
    
    Weights emphasize prompt adherence and aesthetics,
    as these drive dataset quality.
    """
    return (
        ScoringWeights.PROMPT_ADHERENCE * scores.get("prompt_adherence", 0) +
        ScoringWeights.AESTHETIC_QUALITY * scores.get("aesthetic_quality", 0) +
        ScoringWeights.TEXT_CORRECTNESS * scores.get("text_correctness", 0) +
        ScoringWeights.BRAND_ALIGNMENT * scores.get("brand_alignment", 0)
    )

# Example
example_scores = {
    "prompt_adherence": 9.0,
    "aesthetic_quality": 8.5,
    "text_correctness": 7.0,
    "brand_alignment": 8.0,
}
print(f"Composite score: {compute_composite_score(example_scores):.2f}")
```

---

## Part D: Dataset Generation Loop

### D1: Define Controlled Subjects

```python
# These are your dataset subjects
# Keep this curated and small
SUBJECTS = [
    "Minimalist coffee brand poster with clean typography and steam",
    "Modern eco-friendly packaging design for luxury chocolate",
    "Contemporary tech startup office entrance with glass and wood",
    "Artisanal bakery storefront with warm lighting and fresh bread display",
    "Sustainable fashion brand lookbook with neutral tones and natural fabrics",
    "Boutique hotel lobby with marble and soft ambient lighting",
    "Organic skincare product flat lay with natural ingredients",
    "Furniture showroom displaying minimalist wooden pieces",
    "Plant-based restaurant menu photography with vibrant ingredients",
    "Luxury watch brand catalog shot with professional photography",
]

# For quick testing, use subset
SUBJECTS_TEST = SUBJECTS[:3]

logger.info(f"Using {len(SUBJECTS)} subjects for full run")
logger.info(f"Using {len(SUBJECTS_TEST)} subjects for test run")
```

### D2: Prompt Optimization with DSPy

```python
class PromptOptimizer(dspy.Module):
    """
    Uses DSPy to optimize base subject into detailed generation prompt.
    
    This can be simple (concatenation + templates) or
    complex (few-shot with examples, iterative refinement).
    
    For this tutorial: simple template-based approach.
    """
    
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought("subject -> optimized_prompt")
    
    def forward(self, subject: str) -> str:
        """
        Optimize a subject into a detailed, generation-ready prompt.
        
        Args:
            subject: High-level subject (e.g., "Minimalist coffee brand poster")
        
        Returns:
            Detailed prompt optimized for FLUX generation
        """
        # For this tutorial, use deterministic enrichment
        # In production, you could use dspy.ChainOfThought + LLM
        
        enrichments = {
            "style": "professional, high-quality product photography",
            "lighting": "soft natural lighting, studio-grade",
            "details": "sharp focus, vibrant colors, clean composition",
            "format": "4K, magazine-quality, modern aesthetic"
        }
        
        optimized = f"{subject}, {enrichments['style']}, {enrichments['lighting']}, {enrichments['details']}, {enrichments['format']}"
        
        return optimized

prompt_optimizer = PromptOptimizer()
```

### D3: Single Subject Generation Loop

```python
def generate_candidates_for_subject(
    subject: str,
    num_candidates: int = 4,
    max_retries: int = 2
) -> List[Tuple[Image.Image, Dict[str, float], str]]:
    """
    For a single subject:
    1. Optimize prompt
    2. Generate N candidates
    3. Grade each
    4. Return (image, scores, prompt) tuples
    
    Args:
        subject: Subject string
        num_candidates: Number of images to generate per subject
        max_retries: Retries on generation failure
    
    Returns:
        List of (image, scores_dict, prompt) tuples
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject: {subject}")
    logger.info(f"{'='*60}")
    
    # Step 1: Optimize prompt
    optimized_prompt = prompt_optimizer(subject=subject)
    logger.info(f"Optimized prompt: {optimized_prompt}")
    
    candidates = []
    
    # Step 2: Generate candidates
    for i in range(num_candidates):
        logger.info(f"  Generating candidate {i+1}/{num_candidates}...")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Generate image
                image = flux_smashed(
                    prompt=optimized_prompt,
                    num_inference_steps=20,  # Reduced for smashed model
                    guidance_scale=3.5,
                    height=768,
                    width=768,
                    seed=42 + i  # Deterministic for reproducibility
                ).images[0]
                
                break
            except Exception as e:
                retry_count += 1
                logger.warning(f"    Generation failed (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    logger.error(f"    Skipping candidate after {max_retries} retries")
                    image = None
        
        if image is None:
            continue
        
        # Step 3: Grade image
        logger.info(f"    Grading image...")
        try:
            scores = vlm_grader(prompt=optimized_prompt, image=image)
            composite_score = compute_composite_score(scores)
            
            scores["composite"] = composite_score
            logger.info(f"    Composite score: {composite_score:.2f}/10.0")
            logger.info(f"      - Prompt adherence: {scores['prompt_adherence']:.1f}")
            logger.info(f"      - Aesthetic: {scores['aesthetic_quality']:.1f}")
            logger.info(f"      - Text: {scores['text_correctness']:.1f}")
            logger.info(f"      - Brand: {scores['brand_alignment']:.1f}")
            
            candidates.append((image, scores, optimized_prompt))
        
        except Exception as e:
            logger.error(f"    Grading failed: {e}")
            continue
    
    return candidates
```

### D4: Full Dataset Generation Pipeline

```python
def generate_full_dataset(
    subjects: List[str],
    candidates_per_subject: int = 4,
    keep_per_subject: int = 2,
    output_dir: Path = OUTPUT_DIR
) -> List[Dict]:
    """
    Full pipeline:
    1. For each subject:
       - Generate N candidates
       - Grade each
       - Keep top K
    2. Save images and metadata
    
    Args:
        subjects: List of subject strings
        candidates_per_subject: Number to generate per subject
        keep_per_subject: Number to keep per subject (top-k)
        output_dir: Where to save results
    
    Returns:
        List of dataset records (metadata)
    """
    logger.info(f"\n{'#'*60}")
    logger.info("STARTING DATASET GENERATION PIPELINE")
    logger.info(f"{'#'*60}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Candidates per subject: {candidates_per_subject}")
    logger.info(f"Keep per subject: {keep_per_subject}")
    logger.info(f"Expected output: {len(subjects) * keep_per_subject} images")
    
    all_records = []
    image_counter = 0
    
    for subject_idx, subject in enumerate(subjects, 1):
        logger.info(f"\n[{subject_idx}/{len(subjects)}]")
        
        # Generate candidates
        candidates = generate_candidates_for_subject(
            subject=subject,
            num_candidates=candidates_per_subject
        )
        
        if not candidates:
            logger.warning(f"No valid candidates for subject: {subject}")
            continue
        
        # Sort by composite score (descending)
        candidates.sort(key=lambda x: x[1]["composite"], reverse=True)
        
        # Keep top K
        selected = candidates[:keep_per_subject]
        logger.info(f"Selected {len(selected)} / {len(candidates)} candidates")
        
        # Save and record
        for rank, (image, scores, optimized_prompt) in enumerate(selected, 1):
            image_id = f"subject_{subject_idx:02d}_rank_{rank}"
            image_path = IMAGES_DIR / f"{image_id}.png"
            
            # Save image
            image.save(str(image_path))
            logger.info(f"  Saved: {image_path.name}")
            
            # Create record
            record = {
                "image_id": image_id,
                "subject": subject,
                "original_subject": subject,
                "optimized_prompt": optimized_prompt,
                "image_path": str(image_path.relative_to(OUTPUT_DIR)),
                "scores": {
                    "prompt_adherence": float(scores["prompt_adherence"]),
                    "aesthetic_quality": float(scores["aesthetic_quality"]),
                    "text_correctness": float(scores["text_correctness"]),
                    "brand_alignment": float(scores["brand_alignment"]),
                    "composite": float(scores["composite"]),
                },
                "rank_in_subject": rank,
            }
            
            all_records.append(record)
            image_counter += 1
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"✓ PIPELINE COMPLETE")
    logger.info(f"Total images generated: {image_counter}")
    logger.info(f"{'#'*60}\n")
    
    return all_records
```

### D5: Run Generation (Full Run)

```python
def run_full_pipeline():
    """Execute the full dataset generation pipeline"""
    
    # Generate dataset
    records = generate_full_dataset(
        subjects=SUBJECTS,
        candidates_per_subject=4,
        keep_per_subject=2,  # Keep top 2 per subject → ~20 images
        output_dir=OUTPUT_DIR
    )
    
    # Save metadata
    save_metadata(records, METADATA_PATH)
    
    return records

# Uncomment to run (takes 30-60 mins depending on GPU)
# records = run_full_pipeline()
```

### D5b: Run Generation (Test Run)

```python
def run_test_pipeline():
    """
    Quick test run with fewer subjects.
    Use this to validate the full pipeline before running production.
    """
    logger.info("\n⚠️  RUNNING TEST PIPELINE")
    logger.info("Using 3 subjects, 2 candidates each, keep 1 per subject")
    logger.info("Expected: ~3 images in 10-15 minutes\n")
    
    records = generate_full_dataset(
        subjects=SUBJECTS_TEST,
        candidates_per_subject=2,
        keep_per_subject=1,
        output_dir=OUTPUT_DIR
    )
    
    save_metadata(records, METADATA_PATH)
    
    return records

# Run test first
test_records = run_test_pipeline()
```

---

## Part E: Export and Analyze

### E1: Save Metadata

```python
def save_metadata(records: List[Dict], path: Path):
    """Save dataset metadata to JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    
    logger.info(f"✓ Metadata saved to {path}")
```

### E2: Analyze Generated Dataset

```python
def analyze_dataset(records: List[Dict]):
    """Print analysis of generated dataset"""
    
    if not records:
        logger.warning("No records to analyze")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("DATASET ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Aggregate scores
    scores_dict = {
        "prompt_adherence": [],
        "aesthetic_quality": [],
        "text_correctness": [],
        "brand_alignment": [],
        "composite": []
    }
    
    for record in records:
        for key, val in record["scores"].items():
            scores_dict[key].append(val)
    
    # Print stats
    logger.info(f"\nTotal images: {len(records)}")
    logger.info(f"Unique subjects: {len(set(r['subject'] for r in records))}")
    
    logger.info(f"\nScore Statistics (mean ± std):")
    for metric, values in scores_dict.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        logger.info(f"  {metric:20s}: {mean_val:.2f} ± {std_val:.2f} (range: {min_val:.1f}-{max_val:.1f})")
    
    # Score distribution
    logger.info(f"\nComposite Score Distribution:")
    composite_scores = scores_dict["composite"]
    ranges = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 10)]
    for low, high in ranges:
        count = sum(1 for s in composite_scores if low <= s < high)
        pct = 100 * count / len(composite_scores)
        bar = "█" * int(pct / 2)
        logger.info(f"  {low}-{high}: {count:3d} ({pct:5.1f}%) {bar}")
    
    logger.info(f"\n{'='*60}\n")

# Analyze after generation
analyze_dataset(test_records)
```

### E3: Visualize Top Images

```python
def show_top_images(records: List[Dict], top_k: int = 4):
    """
    Display top K images by composite score.
    (Requires Jupyter or matplotlib display)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")
        return
    
    # Sort by composite score
    sorted_records = sorted(
        records,
        key=lambda x: x["scores"]["composite"],
        reverse=True
    )
    
    top_records = sorted_records[:top_k]
    
    fig, axes = plt.subplots(1, top_k, figsize=(5*top_k, 5))
    if top_k == 1:
        axes = [axes]
    
    for ax, record in zip(axes, top_records):
        img_path = OUTPUT_DIR / record["image_path"]
        img = Image.open(img_path)
        
        ax.imshow(img)
        ax.set_title(
            f"Score: {record['scores']['composite']:.2f}\n{record['subject'][:30]}",
            fontsize=10
        )
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# Visualize (in Jupyter)
# show_top_images(test_records, top_k=3)
```

### E4: Export to Hugging Face Hub

```python
def push_to_huggingface(
    records: List[Dict],
    dataset_name: str = "flux-optimized-mini-dataset",
    repo_id: str = None,
    private: bool = True
):
    """
    Export generated dataset to Hugging Face Hub.
    
    Args:
        records: Dataset records
        dataset_name: Local dataset name
        repo_id: HF Hub repo ID (e.g., "your-org/flux-optimized-mini")
        private: Whether to make repo private
    """
    logger.info(f"Preparing dataset for HF Hub export...")
    
    # Build dataset from records
    hf_dataset = datasets.Dataset.from_dict({
        "image_id": [r["image_id"] for r in records],
        "subject": [r["subject"] for r in records],
        "optimized_prompt": [r["optimized_prompt"] for r in records],
        "composite_score": [r["scores"]["composite"] for r in records],
        "prompt_adherence": [r["scores"]["prompt_adherence"] for r in records],
        "aesthetic_quality": [r["scores"]["aesthetic_quality"] for r in records],
        "text_correctness": [r["scores"]["text_correctness"] for r in records],
        "brand_alignment": [r["scores"]["brand_alignment"] for r in records],
    })
    
    # Add image column
    def load_image(image_path):
        return Image.open(OUTPUT_DIR / image_path)
    
    hf_dataset = hf_dataset.map(
        lambda x: {"image": load_image(records[0]["image_path"])},
        load_from_cache_file=False
    )
    
    # Push to hub
    if repo_id:
        logger.info(f"Pushing to {repo_id}...")
        try:
            hf_dataset.push_to_hub(
                repo_id,
                private=private
            )
            logger.info(f"✓ Dataset pushed to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
    else:
        logger.info("repo_id not specified, skipping Hub push")

# Example (requires HF token)
# push_to_huggingface(
#     test_records,
#     repo_id="your-org/flux-optimized-mini-dataset",
#     private=True
# )
```

---

## Optional: Fine-Tuning Setup

### Fine-Tune Motivation

The (original_prompt → best_image) pairs form a **synthetic alignment dataset**:

- **Original**: High-level subject description
- **Best Image**: VLM-approved best generation
- **Interpretation**: This is pseudo-ground-truth for what "good" looks like

This can be used to fine-tune FLUX (or a classifier) to:
1. Improve generation quality for similar subjects
2. Learn aesthetic preferences from VLM scores
3. Build custom image generators

### Fine-Tune Data Format

```python
def prepare_finetuning_dataset(records: List[Dict]):
    """
    Prepare data for fine-tuning.
    
    Format: (subject, optimized_prompt, image, scores)
    """
    finetune_data = []
    
    for record in records:
        finetune_data.append({
            "original_subject": record["subject"],
            "optimized_prompt": record["optimized_prompt"],
            "image_path": str(OUTPUT_DIR / record["image_path"]),
            "composite_score": record["scores"]["composite"],
            # Individual dimensions for weighted loss
            "prompt_adherence": record["scores"]["prompt_adherence"],
            "aesthetic_quality": record["scores"]["aesthetic_quality"],
            "text_correctness": record["scores"]["text_correctness"],
            "brand_alignment": record["scores"]["brand_alignment"],
        })
    
    return finetune_data

finetune_data = prepare_finetuning_dataset(test_records)
logger.info(f"Prepared {len(finetune_data)} samples for fine-tuning")
```

### Fine-Tuning Example (Conceptual)

```python
# This is pseudo-code for fine-tuning direction
# Actual implementation depends on your fine-tuning framework

"""
Example: DreamBooth or LoRA fine-tuning

from peft import LoraConfig, get_peft_model

# Load base FLUX model
model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")

# Create LoRA adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# Fine-tune on (subject → prompt) pairs with composite score as weight
# This teaches FLUX to improve generation quality for your domain

for epoch in range(5):
    for example in finetune_data:
        subject = example["original_subject"]
        prompt = example["optimized_prompt"]
        weight = example["composite_score"] / 10.0  # Normalize to [0,1]
        
        loss = model.train_step(
            prompt=prompt,
            weight=weight
        )
"""
```

---

## Complete Quick-Start Code

```python
# ============================================================================
# QUICK START: Copy-paste this to run the full pipeline
# ============================================================================

from pathlib import Path
from pruna import smash, SmashConfig
from diffusers import FluxPipeline
import dspy
import torch

# 1. Load and optimize model
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

smash_cfg = SmashConfig(batch_size=1, device=device)
smash_cfg.add("deepcache")
smash_cfg.add("torch_compile")

flux_smashed = smash(base_model, smash_cfg)

# 2. Configure VLM
dspy.settings.configure(
    lm=dspy.HFModel(
        model="HuggingFaceTB/SmolVLM-Instruct",
        temperature=0.0
    )
)

# 3. Run generation
subjects = [
    "Minimalist coffee brand poster",
    "Modern eco-friendly packaging",
    "Tech startup office"
]

records = generate_full_dataset(
    subjects=subjects,
    candidates_per_subject=4,
    keep_per_subject=2
)

# 4. Analyze
analyze_dataset(records)

# 5. Export
push_to_huggingface(
    records,
    repo_id="your-org/flux-dataset",
    private=True
)
```

---

## Summary: What This System Does

```
Input: 10 curated subjects
   ↓
DSPy: Enrich each subject → detailed prompts
   ↓
FLUX (Pruna-optimized): Generate 4 candidates per subject
   ↓
SmolVLM: Grade each image on 4 dimensions
   ↓
Top-K Selector: Keep best 2 per subject
   ↓
Output: 20 high-quality, curated, scored images
```

### Key Strengths

✅ **Reproducible**: Seeds, deterministic scoring  
✅ **Controlled**: Only ~10 subjects, all curated  
✅ **Efficient**: Pruna optimization makes it affordable  
✅ **Interpretable**: Per-image scores from VLM  
✅ **Extendable**: Easy to add new subjects or metrics  
✅ **Production-ready**: Metadata, versioning, HF integration  

### Next Steps

1. **Test**: Run `run_test_pipeline()` first (~15 min)
2. **Validate**: Check `analyze_dataset()` output and visualizations
3. **Scale**: Run `run_full_pipeline()` with full SUBJECTS list
4. **Fine-tune** (optional): Use records to train custom FLUX adapter
5. **Deploy**: Push to HF Hub or use for downstream tasks

---

## References

- [Pruna Documentation](https://docs.pruna.ai)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)