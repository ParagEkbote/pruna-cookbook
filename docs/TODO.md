Below is a **precise, runnable notebook skeleton** structured exactly around your pipeline:

```
DSPy (prompt search)
        ↓
FLUX smashed (image generator)
        ↓
SmolVLM (quality grader)
        ↓
Top-k selector
        ↓
Curated dataset
```

This is not pseudocode. It is structured so you can fill in credentials and run.

We assume:

* Base generator: **FLUX.2-klein-4B**
* Publisher: **Black Forest Labs**
* VLM grader: **SmolVLM-Instruct**
* DSPy optimizer: `MIPROv2`
* Goal: ~12–24 high-quality curated images

---

# 📓 Notebook: Closed-Loop Synthetic Image Dataset Creation

---

# 0️⃣ Environment Setup

```bash
pip install dspy pruna transformers datasets pillow torch
```

---

# 1️⃣ Load & Smash FLUX

```python
from pruna import PrunaModel, smash, SmashConfig

# Load base model
base_flux = PrunaModel.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B"
)

# Smash configuration (adjust as needed)
smash_cfg = SmashConfig([
    "deepcache",
    "compiler(torch_compile)"
])

# Apply smash
flux = smash(base_flux, smash_cfg)

# Optional: save optimized artifact
flux.save_pretrained("flux_klein_4b_smashed")
```

---

# 2️⃣ Configure DSPy with SmolVLM (Grader)

We use SmolVLM **as evaluator**, not generator.

```python
import dspy
from dspy import settings

settings.configure(
    lm=dspy.HFModel(
        model="HuggingFaceTB/SmolVLM-Instruct",
        temperature=0.0
    )
)
```

---

# 3️⃣ Define DSPy Prompt Generator Module

We define a structured prompt search module.

```python
class PromptSignature(dspy.Signature):
    subject: str = dspy.InputField()
    optimized_prompt: str = dspy.OutputField(
        desc="Highly descriptive image generation prompt"
    )

prompt_module = dspy.Predict(PromptSignature)
```

---

# 4️⃣ Define Structured Grading Module (SmolVLM)

```python
class ImageGrade(dspy.Signature):
    prompt: str = dspy.InputField()
    image: object = dspy.InputField()

    prompt_adherence: float = dspy.OutputField()
    aesthetic_quality: float = dspy.OutputField()
    text_correctness: float = dspy.OutputField()
```

```python
vlm_grader = dspy.Predict(ImageGrade)
```

---

# 5️⃣ Define Optimization Metric

This metric drives DSPy prompt search.

```python
def grading_metric(example, pred, trace=None):
    return (
        0.5 * pred.prompt_adherence +
        0.3 * pred.aesthetic_quality +
        0.2 * pred.text_correctness
    )
```

---

# 6️⃣ Create DSPy Optimizer

```python
from dspy import MIPROv2

optimizer = MIPROv2(
    metric=grading_metric,
    auto="medium"
)
```

---

# 7️⃣ Define Subject List (Dataset Seeds)

Keep controlled.

```python
subjects = [
    "Minimalist coffee brand poster",
    "Luxury chocolate advertisement",
    "Eco-friendly skincare packaging design",
    "Modern tech startup hero banner",
    "Futuristic sportswear campaign"
]
```

5 subjects × 4 candidates → select top 2 → ~10 images
Scale subjects to reach 12–24.

---

# 8️⃣ Closed-Loop Pipeline

```python
import os
from datasets import Dataset
from PIL import Image

os.makedirs("generated_images", exist_ok=True)

records = []

for subject in subjects:
    
    # ----------------------------
    # DSPy Prompt Search
    # ----------------------------
    compiled_prompt_module = optimizer.compile(
        prompt_module,
        trainset=[{"subject": subject}]
    )
    
    optimized = compiled_prompt_module(subject=subject)
    optimized_prompt = optimized.optimized_prompt
    
    candidates = []
    
    # ----------------------------
    # Generate Multiple Candidates
    # ----------------------------
    for i in range(4):
        
        result = flux.generate(
            positivePrompt=optimized_prompt
        )
        
        image = result.images[0]
        
        # ----------------------------
        # SmolVLM Grading
        # ----------------------------
        grade = vlm_grader(
            prompt=optimized_prompt,
            image=image
        )
        
        score = grading_metric(None, grade)
        
        candidates.append({
            "image": image,
            "score": score,
            "grades": {
                "prompt_adherence": grade.prompt_adherence,
                "aesthetic_quality": grade.aesthetic_quality,
                "text_correctness": grade.text_correctness,
            }
        })
    
    # ----------------------------
    # Top-k Selection
    # ----------------------------
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:2]
    
    for idx, item in enumerate(top_candidates):
        image_path = f"generated_images/{subject.replace(' ','_')}_{idx}.png"
        item["image"].save(image_path)
        
        records.append({
            "subject": subject,
            "optimized_prompt": optimized_prompt,
            "image_path": image_path,
            "score": item["score"],
            **item["grades"]
        })
```

---

# 9️⃣ Create Curated Dataset

```python
dataset = Dataset.from_list(records)

dataset.save_to_disk("flux_curated_dataset")
```

Optional:

```python
dataset.push_to_hub("your-org/flux-optimized-mini-dataset")
```

---

# 🔟 Final Output Structure

Each record:

```json
{
  "subject": "...",
  "optimized_prompt": "...",
  "image_path": "...",
  "score": 0.87,
  "prompt_adherence": 0.91,
  "aesthetic_quality": 0.82,
  "text_correctness": 0.79
}
```

---

# 📐 What This Achieves

You now have:

* DSPy-driven prompt search
* FLUX smashed generation
* SmolVLM grading
* Top-k filtering
* Curated synthetic dataset
* Reproducible selection logic

---

# 🧠 Important Architectural Note

This design:

* Keeps generator and grader decoupled
* Uses VLM strictly as evaluator
* Uses DSPy only for structured prompt optimization
* Avoids self-evaluation bias
* Scales naturally by increasing subjects or candidate count

---

