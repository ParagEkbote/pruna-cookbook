# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#     pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
#     "unsloth",
#     "transformers>=4.40",
#     "datasets",
#     "accelerate",
#     "peft",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
# ]
# ///

"""
UnSloth LoRA SFT Training Script
"""

import argparse
import logging
import os
import sys
import time
import traceback
import random

import torch
import torch.nn as nn
import numpy as np

from typing import List

from huggingface_hub import login, HfApi
from datasets import load_dataset

from transformers import (
    default_data_collator,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model_state_dict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_cuda():
    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"GPU: {name}")
    logger.info(f"Memory: {mem:.1f} GB")


# ==========================================================
# Training Monitoring
# ==========================================================

class TrainingMonitor:

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.losses = []
        self.warnings = []

    def add_loss(self, loss: float):

        self.losses.append(loss)

        if len(self.losses) > self.window_size:

            recent = self.losses[-self.window_size:]

            mean = np.mean(recent)
            std = np.std(recent)

            if std > 0 and abs(loss - mean) > 3 * std:
                self.warnings.append(
                    f"Loss spike {loss:.4f} (mean {mean:.4f}, std {std:.4f})"
                )

    def summary(self):

        if not self.losses:
            return {}

        return dict(
            steps=len(self.losses),
            avg_loss=float(np.mean(self.losses)),
            warnings=len(self.warnings),
            recent_warnings=self.warnings[-5:],
        )


# ==========================================================
# LoRA target detection
# ==========================================================

def get_target_modules(model) -> List[str]:

    targets = []

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            if any(
                key in name
                for key in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                targets.append(name)

    logger.info(f"LoRA applied to {len(targets)} modules")

    return targets


# ==========================================================
# Argument parsing
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-repo", required=True)

    parser.add_argument(
        "--base-model",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--merge-lora", action="store_true")

    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--save-steps", type=int, default=500)

    parser.add_argument("--save-local", default="output")

    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--no-push-hub", action="store_true")

    return parser.parse_args()


# ==========================================================
# Training
# ==========================================================

def main():

    args = parse_args()

    try:

        logger.info("Starting training")

        set_seed(args.seed)

        check_cuda()

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        token = os.environ.get("HF_TOKEN")

        if token:
            login(token)

        # ===============================
        # Load model
        # ===============================

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_16bit=True,
        )

        model.config.use_cache = False

        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        target_modules = get_target_modules(model)

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )

        device = next(model.parameters()).device

        # ===============================
        # Dataset
        # ===============================

        dataset = load_dataset(args.dataset, split="train")

        assert "conversations" in dataset.features

        def format_chat(examples):

            text = tokenizer.apply_chat_template(
                examples["conversations"],
                tokenize=False,
            )

            return {"text": text}

        dataset = dataset.map(
            format_chat,
            remove_columns=dataset.column_names,
        )

        # ===============================
        # Tokenization
        # ===============================

        def tokenize_function(examples):

            out = tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            )

            out["labels"] = out["input_ids"].copy()

            return out

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        tokenized_dataset.set_format(type="torch")

        # ===============================
        # DataLoader
        # ===============================

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )

        # ===============================
        # Optimizer
        # ===============================

        lora_params = [
            p for n, p in model.named_parameters()
            if n in get_peft_model_state_dict(model)
        ]

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=args.learning_rate,
            fused=True,
        )

        updates_per_epoch = len(dataloader) // args.gradient_accumulation

        total_steps = updates_per_epoch * args.num_epochs

        warmup = int(0.05 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        # ===============================
        # Precision
        # ===============================

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        scaler = (
            torch.cuda.amp.GradScaler()
            if dtype == torch.float16
            else None
        )

        # ===============================
        # Training
        # ===============================

        monitor = TrainingMonitor()

        optimizer_step = 0

        start = time.time()

        model.train()

        for epoch in range(args.num_epochs):

            for step, batch in enumerate(dataloader):

                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)

                with torch.cuda.amp.autocast(dtype=dtype):

                    out = model(
                        input_ids=input_ids,
                        attention_mask=mask,
                        labels=input_ids,
                    )

                    loss = out.loss / args.gradient_accumulation

                if scaler:

                    scaler.scale(loss).backward()

                else:

                    loss.backward()

                if (step + 1) % args.gradient_accumulation == 0:

                    if scaler:

                        scaler.step(optimizer)
                        scaler.update()

                    else:

                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                    optimizer_step += 1

                    loss_val = loss.item() * args.gradient_accumulation

                    monitor.add_loss(loss_val)

                    if optimizer_step % 50 == 0:

                        elapsed = time.time() - start

                        logger.info(
                            f"Step {optimizer_step}/{total_steps} "
                            f"Loss {loss_val:.4f} "
                            f"Time {elapsed/60:.1f}m"
                        )

                    if optimizer_step % args.save_steps == 0:

                        logger.info("Checkpoint saving")

                        model.save_pretrained(
                            f"{args.save_local}/checkpoint-{optimizer_step}"
                        )

        summary = monitor.summary()

        logger.info(summary)

        torch.cuda.empty_cache()

        # ===============================
        # Save
        # ===============================

        logger.info("Saving final model")

        model_to_save = model

        if args.merge_lora and hasattr(model, "merge_and_unload"):
            model_to_save = model.merge_and_unload()

        model_to_save.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)

        # ===============================
        # Push
        # ===============================

        if not args.no_push_hub:

            api = HfApi()

            api.upload_folder(
                folder_path=args.save_local,
                repo_id=args.output_repo,
                repo_type="model",
            )

            logger.info(f"Pushed to {args.output_repo}")

        logger.info("Training complete")

    except KeyboardInterrupt:

        logger.info("Interrupted")

    except Exception as e:

        logger.error(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()