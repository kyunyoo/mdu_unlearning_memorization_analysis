"""
Fine-tune LLaDA-8B on TOFU (forget10 split) using the masked diffusion loss.

Saves a checkpoint every --save_every_n_epochs epochs so that memorization
can be tracked over training (e.g., at 10, 100, 200, ..., 1000 epochs).

Usage:
    python scripts/finetune_tofu.py \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --output_dir checkpoints/ \
        --n_epochs 1000 \
        --save_every_n_epochs 100 \
        --batch_size 4 \
        --grad_accum 4 \
        --lr 2e-5

Saves checkpoints as:
    checkpoints/ep10/
    checkpoints/ep100/
    checkpoints/ep200/
    ...
    checkpoints/ep1000/
"""

import os
import sys
import math
import json
import argparse
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASK_TOKEN_ID = 126336  # LLaDA <|mdm_mask|>
IGNORE_INDEX = -100


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TOFUDataset(Dataset):
    """TOFU QA tokenized for DLM training.

    Returns input_ids with labels=-100 at prompt positions and token ids at
    answer positions (standard causal-LM label convention).
    """
    def __init__(self, tokenizer, split: str = "full", max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = load_dataset("locuslab/TOFU", split, split="train")
        self.samples = [{"question": r["question"], "answer": r["answer"]} for r in raw]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q = self.samples[idx]["question"]
        a = self.samples[idx]["answer"]

        messages = [{"role": "user", "content": q}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        answer_ids = self.tokenizer(a, add_special_tokens=False).input_ids
        full_ids = (prompt_ids + answer_ids)[: self.max_length]

        labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids)) + \
                 full_ids[len(prompt_ids):]
        labels = labels[: self.max_length]

        # Pad to same length inside collate_fn, so just return tensors here
        ids_t = torch.tensor(full_ids, dtype=torch.long)
        lbl_t = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": ids_t, "labels": lbl_t,
                "attention_mask": torch.ones_like(ids_t)}


def collate_fn(batch, pad_id: int = 0):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels     = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L]  = b["input_ids"]
        labels[i, :L]     = b["labels"]
        attn_mask[i, :L]  = b["attention_mask"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ─────────────────────────────────────────────────────────────────────────────
# DLM training loss  (Eq.14 training objective, same as eval's ELBO proxy)
# ─────────────────────────────────────────────────────────────────────────────

def dlm_loss(model, batch, device: str = "cuda", eps: float = 1e-3):
    """
    Standard masked-diffusion training loss for LLaDA.

    Steps:
      1. Sample t ~ U(0,1) per sample  →  p_mask = (1 - eps) * t + eps
      2. Randomly replace each answer token with <mask> w.p. p_mask
      3. Forward pass on noisy input_ids
      4. CE loss at masked answer positions, weighted by 1/p_mask (ELBO)

    This is the *same* objective as Eq.(14) used during evaluation,
    so probability at eval time directly reflects training objective.
    """
    input_ids    = batch["input_ids"].to(device)
    labels       = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    B, L = input_ids.shape
    answer_mask = (labels != IGNORE_INDEX)  # [B, L]

    t = torch.rand(B, device=device)
    p = (1 - eps) * t + eps                 # [B]
    p_exp = p[:, None].expand(B, L)         # [B, L]

    rand = torch.rand_like(p_exp)
    masked_indices = (rand < p_exp) & answer_mask  # [B, L]

    noisy = input_ids.clone()
    noisy[masked_indices] = MASK_TOKEN_ID

    out = model(input_ids=noisy, attention_mask=attention_mask)
    logits = out.logits                      # [B, L, V]
    V = logits.shape[-1]

    ce = F.cross_entropy(
        logits.view(-1, V), input_ids.view(-1),
        reduction="none", ignore_index=IGNORE_INDEX,
    ).view(B, L)                             # [B, L]

    ce = ce * masked_indices.float()
    weight = (1.0 / p_exp.clamp(min=1e-6))  # ELBO correction

    # Average over masked positions per sample, then over batch
    n_masked = masked_indices.float().sum(dim=-1).clamp(min=1)
    loss = ((ce * weight).sum(dim=-1) / n_masked).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # copy modeling_llada.py so checkpoint is self-contained for eval
    llada_src = None
    for search in [
        os.path.join(os.path.dirname(model.config._name_or_path), "modeling_llada.py"),
        os.path.join(REPO_ROOT, "modeling_llada.py"),
    ]:
        if os.path.isfile(search):
            llada_src = search
            break
    if llada_src:
        import shutil
        shutil.copy(llada_src, out_dir)
    print(f"  [ckpt] saved → {out_dir}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = MASK_TOKEN_ID

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()

    dataset = TOFUDataset(tokenizer, split=args.tofu_split, max_length=args.max_length)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id or 0),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch  = math.ceil(len(dataset) / args.batch_size / args.grad_accum)
    total_steps      = steps_per_epoch * args.n_epochs
    warmup_steps     = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nDataset: {len(dataset)} samples  |  {steps_per_epoch} opt-steps/epoch")
    print(f"Total opt-steps: {total_steps}  |  Warmup: {warmup_steps}")
    print(f"Save every {args.save_every_n_epochs} epochs\n")

    # Always save epoch 10 for the baseline memorization point
    save_epochs = set(args.extra_save_epochs or [])
    for ep in range(args.save_every_n_epochs, args.n_epochs + 1, args.save_every_n_epochs):
        save_epochs.add(ep)

    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_in_epoch, batch in enumerate(loader):
            loss = dlm_loss(model, batch, device=device) / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step_in_epoch + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:4d}/{args.n_epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch in save_epochs:
            ckpt_dir = os.path.join(args.output_dir, f"ep{epoch}")
            save_checkpoint(model, tokenizer, ckpt_dir)

    print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct",
                   help="Base model (HF id or local path)")
    p.add_argument("--output_dir", default="checkpoints/",
                   help="Root dir; checkpoints saved as ep<N>/")
    p.add_argument("--tofu_split", default="full",
                   help="TOFU split to fine-tune on (full / forget10)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=1000)
    p.add_argument("--save_every_n_epochs", type=int, default=100,
                   help="Save checkpoint at multiples of this value")
    p.add_argument("--extra_save_epochs", nargs="*", type=int, default=[10],
                   help="Additional epochs to save (default: 10)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
