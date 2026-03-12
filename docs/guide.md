# Controlled Synthetic Dataset Generation: Complete Guide

## Executive Summary

This system generates **12–24 high-quality, curated images** through a closed-loop pipeline:

```
Subjects (10-12)
    ↓
DSPy Prompt Optimization
    ↓
FLUX.2-klein-4B (Pruna-optimized) → 4 candidates per subject
    ↓
SmolVLM Grading → 4 dimensions scored
    ↓
Top-K Selection → Keep 2 best per subject
    ↓
Final Dataset: ~20 images with full provenance
```

**Key Innovation**: VLM is used purely as an **evaluator**, not a generator. This ensures:
- ✅ Reproducible results
- ✅ Explicit quality signals
- ✅ Clean separation of concerns
- ✅ Extendable metric framework

---

## System Architecture

### 1️⃣ Layer 1: Model Optimization (Pruna)

**Goal**: Make inference fast enough for iterative loops

**How it works**:
- Load FLUX.2-klein-4B (4B parameters, lightweight)
- Apply Pruna SmashConfig with:
  - **DeepCache**: Reuse transformer outputs across diffusion steps
  - **Torch Compile**: Compile to optimized kernels (Inductor backend)

**Expected gains**:
- ~2-3x latency reduction
- ~10-15% memory reduction
- Enables multiple candidate generations per subject

**Code reference**:
```python
smash_cfg = SmashConfig()
smash_cfg.add("deepcache")
smash_cfg.add("torch_compile")
flux_smashed = smash(base_model, smash_cfg)
```

**Why FLUX.2-klein-4B?**
- Smaller than FLUX.1 (~67B) or Flux Pro
- Still high-quality output
- Fast enough for 12-24 image generation in minutes, not hours
- Perfect for controlled experiments

---

### 2️⃣ Layer 2: Prompt Optimization (DSPy)

**Goal**: Convert high-level subjects into detailed, generation-ready prompts

**How it works**:

Simple deterministic approach (this tutorial):
```python
enrichments = {
    "style": "professional, high-quality product photography",
    "lighting": "soft natural lighting, studio-grade",
    "details": "sharp focus, vibrant colors, clean composition",
    "format": "4K, magazine-quality, modern aesthetic"
}

optimized = f"{subject}, {enrichments['style']}, ..."
```

Advanced approach (optional):
- Use LLMs to generate multiple candidate prompts
- Score each with a quality metric
- Select best for generation

**Why this layer?**
- Ensures consistency in prompt quality
- Makes results reproducible
- Separates concern of "what to generate" from "how to grade"

---

### 3️⃣ Layer 3: Image Generation (FLUX + Pruna)

**Goal**: Generate multiple candidate images per subject

**Process**:
```
For each subject:
  For candidate in range(4):
    image = flux_smashed(optimized_prompt)
    save_image(image)
```

**Configuration**:
- `num_inference_steps=20` (reduced from default 50, feasible due to Pruna)
- `guidance_scale=3.5` (FLUX likes lower guidance)
- `seed=42+i` (deterministic for reproducibility)
- `height=768, width=768` (balanced size for quality + speed)

**Why multiple candidates?**
- Not all generations are equal even with same prompt
- VLM grading will select the best
- Increases dataset quality through selection

---

### 4️⃣ Layer 4: Quality Grading (SmolVLM)

**Goal**: Score images on multiple quality dimensions

**Why SmolVLM?**
- ~600M parameters (lightweight, fast)
- Multimodal (can see images + text)
- Can be run in low_memory mode on CPU
- Good enough for structured scoring

**Grading Schema**:
```python
class ImageGrade:
    prompt_adherence: float      # Does image match prompt? (0-10)
    aesthetic_quality: float     # Is it visually good? (0-10)
    text_correctness: float      # Are text elements rendered right? (0-10)
    brand_alignment: float       # Does it feel professional? (0-10)
```

**Composite Score**:
```
score = 0.40*adherence + 0.30*aesthetic + 0.20*text + 0.10*brand
```

Weights emphasize:
1. Adherence (most important: does it match the prompt?)
2. Aesthetics (second: does it look good?)
3. Text accuracy (third: is OCR correct?)
4. Brand fit (fourth: professional appearance)

---

### 5️⃣ Layer 5: Selection & Storage

**Goal**: Keep only the best images and store full provenance

**Selection criteria**:
- Sort candidates by composite score (descending)
- Keep top 2 per subject (configurable)

**Stored metadata** (for each image):
```json
{
  "image_id": "subject_01_rank_1",
  "subject": "Minimalist coffee brand poster...",
  "optimized_prompt": "Minimalist coffee brand poster, professional...",
  "image_path": "images/subject_01_rank_1.png",
  "scores": {
    "prompt_adherence": 9.0,
    "aesthetic_quality": 8.5,
    "text_correctness": 7.0,
    "brand_alignment": 8.0,
    "composite": 8.30
  },
  "rank": 1
}
```

---

## Execution Plan

### Phase 1: Setup (5 min)

```bash
# Install dependencies
pip install pruna dspy-ai transformers pillow datasets torch

# Verify HF token (for gated models)
huggingface-cli login
```

### Phase 2: Test Run (15-20 min)

```python
config = GenerationConfig(
    subjects=SUBJECTS[:3],  # Only 3 subjects
    candidates_per_subject=2,
    keep_per_subject=1,
)
```

**Expected output**: ~3 images

**Validates**:
- FLUX loads correctly
- Pruna optimization applies
- DSPy + SmolVLM work
- Grading pipeline works
- Output format is correct

### Phase 3: Production Run (45-90 min)

```python
config = GenerationConfig(
    subjects=SUBJECTS,  # All 10 subjects
    candidates_per_subject=4,
    keep_per_subject=2,
)
```

**Expected output**: ~20 images

**Time breakdown** (rough):
- Model load + optimization: 5 min
- DSPy/SmolVLM setup: 3 min
- Generation loop: 35-80 min (depends on GPU)
  - 10 subjects × 4 candidates × ~1 min per image = 40 min
  - Grading adds ~20% overhead

### Phase 4: Analysis & Export (5 min)

```python
analyze_dataset(records)  # Print statistics
push_to_huggingface(records, repo_id="...")  # Optional
```

---

## Configuration Tuning

### Quick vs. Quality Tradeoff

| Setting | Inference Steps | Guidance | Candidates | Keep | Result |
|---------|-----------------|----------|-----------|------|--------|
| **Fast** | 12 | 2.5 | 2 | 1 | 10 images, 20 min |
| **Balanced** | 20 | 3.5 | 4 | 2 | 20 images, 60 min |
| **Quality** | 30 | 4.5 | 6 | 3 | 30 images, 120 min |

### Memory Considerations

If OOM occurs:
1. Reduce image resolution: `768 → 512`
2. Reduce batch size in SmashConfig: `batch_size=1` (already minimal)
3. Run grading on CPU: `device="cpu"`
4. Reduce candidates per subject: `4 → 2`

### Quality Tuning

To improve average composite score:

1. **Increase inference steps**: 20 → 30
   - Cost: +50% latency
   - Gain: +0.5-1.0 points on scale of 10

2. **Adjust guidance scale**: 3.5 → 4.5
   - Cost: Slightly more artifacts sometimes
   - Gain: Better prompt adherence

3. **Reweight scoring**: Adjust `ScoringWeights`
   - If you care more about aesthetics: increase `AESTHETIC_QUALITY`
   - If text is critical: increase `TEXT_CORRECTNESS`

---

## Expected Results

### Quantitative

After full pipeline with 10 subjects:

```
Total images: 20
Unique subjects: 10
Average composite score: 7.8 ± 1.2

Score breakdown:
  prompt_adherence    : 8.1 ± 1.1
  aesthetic_quality   : 7.9 ± 1.3
  text_correctness    : 7.2 ± 1.8
  brand_alignment     : 7.8 ± 1.0

Score distribution:
  0-3  : 0 (0%)
  3-5  : 1 (5%)
  5-7  : 5 (25%)
  7-9  : 12 (60%)
  9-10 : 2 (10%)
```

Most images in 7-9 range = good quality, some room for improvement.

### Qualitative

- Images are consistent in style
- Text rendering (if present) is usually correct
- Professional, on-brand appearance
- Suitable for product catalogs, marketing
- Reproducible: same subjects → same images (seeds fixed)

---

## Fine-Tuning Integration (Optional)

The (original_subject → best_image) pairs can be used to fine-tune FLUX:

### Why fine-tune?

Your optimized prompts + best images represent your aesthetic preferences.
Fine-tuning teaches FLUX to prefer:
- Your particular style
- Your quality standards
- Your domain (coffee, fashion, tech, etc.)

### Data format

```python
finetune_data = [
    {
        "original_subject": "Minimalist coffee...",
        "optimized_prompt": "Minimalist coffee..., professional, studio...",
        "image_path": "images/subject_01_rank_1.png",
        "composite_score": 8.3,
    },
    ...
]
```

### Fine-tuning approach

**LoRA** (recommended):
- Fast: 30-60 min on single GPU
- Efficient: ~100MB final model
- Extendable: can train multiple LoRAs

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
)

model = get_peft_model(flux, lora_config)

for example in finetune_data:
    loss = model.train_step(
        prompt=example["optimized_prompt"],
        weight=example["composite_score"] / 10.0  # Normalize
    )
```

**DreamBooth** (alternative):
- Preserves FLUX behavior better
- Requires more data (~50+ images)
- Slower but higher quality

---

## Troubleshooting

### Problem: CUDA out of memory

**Solution**:
```python
# Option 1: Reduce resolution
config.image_height = 512
config.image_width = 512

# Option 2: Reduce candidates
config.candidates_per_subject = 2

# Option 3: Disable torch.compile (faster inference but less optimized)
# Remove torch_compile from SmashConfig
```

### Problem: FLUX model won't load

**Solution**:
```bash
# Ensure HF token is set
huggingface-cli login

# Check model exists and is accessible
huggingface-cli model info black-forest-labs/FLUX.2-klein-4B

# Try downloading separately first
from transformers import AutoModel
model = AutoModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B")
```

### Problem: SmolVLM grading is slow

**Solution**:
```python
# Run on CPU (less VRAM intensive)
# In EvaluationTask: low_memory=True

# Or: Batch grade multiple images at once (requires code modification)

# Or: Sample only subset of candidates for grading (trade quality for speed)
```

### Problem: Very low composite scores (< 5)

**Causes**:
- Prompts too vague or ambiguous
- FLUX struggling with domain (text, fine details, etc.)
- SmolVLM being too strict

**Solutions**:
1. Improve prompts: add more detail to subjects
2. Lower guidance scale: 3.5 → 2.5
3. Adjust weights: reduce strict metrics if necessary
4. Use more inference steps: 20 → 30

---

## Cost Analysis

### GPU Time (on NVIDIA A100)

```
Test run (3 subjects, 2 candidates each):
- Setup: 5 min
- Generation: 6 min (3×2×1 min per image)
- Grading: 2 min
- Total: ~13 min

Full run (10 subjects, 4 candidates each, keep 2):
- Setup: 5 min
- Generation: 40 min (10×4×1 min)
- Grading: 8 min
- Total: ~53 min

Cost on cloud GPU (~$0.40/min for A100):
- Test: $5
- Full: $21
```

### Storage

```
Test run (3 images):
- Images (PNG): 3 × 5MB = 15MB
- Metadata: 10KB
- Total: ~15MB

Full run (20 images):
- Images: 20 × 5MB = 100MB
- Metadata: 50KB
- Total: ~100MB
```

---

## Next Steps After Generation

### 1. Validate Results

```python
# Inspect top and bottom images
top_k = sorted(records, key=lambda x: x["scores"]["composite"], reverse=True)[:5]
bottom_k = sorted(records, key=lambda x: x["scores"]["composite"])[:5]

# Visualize (in Jupyter)
show_top_images(top_k)
show_bottom_images(bottom_k)
```

### 2. Iterate Subjects

- Add new subjects → rerun pipeline
- Remove low-scoring subjects
- Refine prompts based on what worked

### 3. Fine-Tune (Optional)

```python
finetune_data = prepare_finetuning_dataset(records)

# Train LoRA adapter on your best images
# This teaches FLUX to match your quality bar
```

### 4. Deploy

- Push to HF Hub for reproducibility
- Use as benchmark dataset
- Fine-tune downstream models
- Generate more images with trained adapter

### 5. Collect Feedback

- Show images to domain experts
- Get human feedback on scores
- Adjust weighting if needed
- Retrain or adjust metrics

---

## Advanced Features (Not Covered Here)

These are possible extensions:

1. **Few-shot prompt optimization**: Use LLM to generate multiple prompts per subject, score them, select best
2. **Iterative refinement**: Use VLM feedback to suggest prompt improvements, regenerate
3. **Multi-model generation**: Generate with FLUX + other models, compare
4. **Dynamic weighting**: Learn weight distribution from human feedback
5. **Negative sampling**: Deliberately generate bad images for contrast learning
6. **Batch optimization**: Use Pruna to find optimal batch size for your GPU
7. **Multi-GPU**: Distribute generation across GPUs for faster execution
8. **Caching**: Cache SmolVLM embeddings to avoid recomputing

---

## Key Insights

### Why This Design?

1. **Separation of Concerns**: Each layer has one responsibility
   - Pruna: speed
   - DSPy: prompt quality
   - FLUX: generation
   - SmolVLM: evaluation

2. **Reproducibility**: Deterministic at every step
   - Seeds fixed
   - Prompts deterministic
   - Grading deterministic (temperature=0.0)
   - Same input → same output

3. **Iterability**: Easy to loop and improve
   - Add subjects → rerun
   - Adjust weights → rerun
   - Improve prompts → rerun
   - All cheap because of Pruna optimization

4. **Interpretability**: Full provenance for each image
   - Know why it was selected
   - Know how it scored
   - Can analyze failure cases

### Why VLM as Evaluator, Not Generator?

**Bad approach**: "VLM generates improved prompts → FLUX generates image"
- Why bad: VLM adds a layer of indirection, confuses "what" with "how"
- Slower: two models (VLM + FLUX)
- Less reproducible: VLM outputs vary even at T=0

**Good approach** (this tutorial): "Deterministic prompt optimization → FLUX generates → VLM grades"
- Why good: Clean pipeline, each stage is focused
- Faster: Only one LLM (SmolVLM for grading, lightweight)
- More reproducible: Everything is deterministic
- More interpretable: You know exactly why each image was selected

---

## Summary Checklist

- [ ] Install dependencies (Pruna, DSPy, transformers, etc.)
- [ ] Verify HF token
- [ ] Run test pipeline (3 subjects, 15 min)
- [ ] Validate outputs (check metadata, images)
- [ ] Run full pipeline (10 subjects, 1 hour)
- [ ] Analyze results (check statistics, visualize)
- [ ] Export to HF Hub (optional)
- [ ] Fine-tune on results (optional)
- [ ] Share dataset with team/community

---

## References

- **Pruna**: https://docs.pruna.ai
- **FLUX.2-klein-4B**: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- **SmolVLM**: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
- **DSPy**: https://github.com/stanfordnlp/dspy
- **Diffusers**: https://huggingface.co/docs/diffusers
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets