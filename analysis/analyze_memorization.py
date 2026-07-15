"""
Memorization analysis for Discrete Diffusion Language Models (DLMs).

Measures RougeL recall and Eq.(14) probability (LLaDA ELBO proxy) on the
TOFU forget set across fine-tuning checkpoints (e.g., 10 / 100 / 200 / ... / 1000 epochs),
and produces a publication-ready plot showing memorization growth over training.

Usage:
    # Single checkpoint
    python analysis/analyze_memorization.py \
        --model_path checkpoints/llada_tofu_ep100 \
        --epoch 100 \
        --output_dir outputs/

    # Sweep mode: multiple checkpoints at once
    python analysis/analyze_memorization.py --sweep \
        --checkpoint_dir checkpoints/ \
        --output_dir outputs/

    # Plot only (from existing outputs/memorization_results.json)
    python analysis/analyze_memorization.py --plot_only \
        --results_json outputs/memorization_results.json \
        --output_dir outputs/
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer

# ── dllm sampler (bundled in this repo under dllm/) ──────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm"))
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

MASK_TOKEN_ID = 126336  # LLaDA mask token


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement functions
# ─────────────────────────────────────────────────────────────────────────────

def build_qa_input(tokenizer, question: str, answer: str, device: str = "cuda"):
    """Tokenize a (Q, A) pair; return full_ids and binary answer_mask."""
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    full_ids = prompt_ids + answer_ids
    answer_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return (
        torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(answer_mask, dtype=torch.long).unsqueeze(0).to(device),
    )


def compute_eq14_probability(
    model, input_ids, answer_mask, n_samples: int = 128, batch_size: int = 64
) -> float:
    """
    Estimate P_θ(answer | question) via Eq.(14) Monte Carlo ELBO (LLaDA App. A.2).

    For a DLM, the standard AR log-likelihood (causal token shift) is *invalid*
    because the model is bidirectional. Instead we use the ELBO:

        log P_θ(x) ≈ E_{l~U[1,L], S~C(L,l)} [L/l · Σ_{i∈S} log p_θ(x_i | x_masked_S)]

    where S is a random subset of answer positions of size l.
    This is the same estimator used in the original LLaDA paper.

    Returns exp(-loss) as a probability proxy (higher = more memorized).
    """
    ans_bool = answer_mask.bool().squeeze(0)
    ans_pos = ans_bool.nonzero(as_tuple=True)[0]
    L = ans_pos.shape[0]
    if L == 0:
        return float("nan")

    # Build n_samples noised copies
    all_noised = input_ids.expand(n_samples, -1).clone()
    mask_flags = torch.zeros(
        n_samples, input_ids.shape[1], device=input_ids.device, dtype=torch.bool
    )
    ls = []
    for s in range(n_samples):
        l = torch.randint(1, L + 1, (1,)).item()
        ls.append(l)
        perm = torch.randperm(L, device=input_ids.device)[:l]
        positions = ans_pos[perm]
        all_noised[s, positions] = MASK_TOKEN_ID
        mask_flags[s, positions] = True

    target_ids = input_ids.expand(n_samples, -1)
    total = 0.0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        with torch.no_grad():
            logits = model(all_noised[start:end]).logits
        token_nll = F.cross_entropy(
            logits.transpose(1, 2), target_ids[start:end], reduction="none"
        )
        for j in range(end - start):
            ce_sum = token_nll[j][mask_flags[start + j]].sum().item()
            total += ce_sum / ls[start + j]

    loss = total / n_samples
    return math.exp(-loss)


def generate_answer(sampler, sampler_config, tokenizer, question: str) -> str:
    """Greedy masked-diffusion generation for a question."""
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
    output = sampler.sample([prompt_tensor], config=sampler_config)
    pred_ids = output.sequences[0][len(prompt_ids):]
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    model_path: str,
    tokenizer_path: str,
    forget_split: str = "forget10",
    n_samples: int = None,
    steps: int = 128,
    max_new_tokens: int = 128,
    mask_samples: int = 64,
    mc_batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Load a checkpoint and measure memorization on the TOFU forget set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = MASK_TOKEN_ID

    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=0.0,
        remasking="low_confidence",
        return_dict=True,
    )

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    forget_ds = load_dataset("locuslab/TOFU", forget_split, split="train")
    n = n_samples or len(forget_ds)

    rougeL_list, prob_list = [], []
    per_sample = []

    for i, sample in enumerate(tqdm(forget_ds, desc="forget", total=n)):
        if i >= n:
            break
        q, a = sample["question"], sample["answer"]

        # RougeL recall
        pred = generate_answer(sampler, sampler_config, tokenizer, q)
        rl = rouge.score(a, pred)["rougeL"].recall
        rougeL_list.append(rl)

        # Eq.(14) probability
        ids, amask = build_qa_input(tokenizer, q, a, device)
        prob = compute_eq14_probability(model, ids, amask, mask_samples, mc_batch_size)
        prob_list.append(prob)

        per_sample.append({"question": q, "answer": a, "pred": pred,
                           "rougeL": rl, "probability": prob})
        print(f"  [{i+1}/{n}] rL={np.mean(rougeL_list):.3f}  prob={np.mean(prob_list):.4f}")

    result = {
        "model_path": model_path,
        "forget_split": forget_split,
        "n": len(rougeL_list),
        "rougeL_mean": float(np.mean(rougeL_list)),
        "rougeL_std": float(np.std(rougeL_list)),
        "probability_mean": float(np.mean(prob_list)),
        "probability_std": float(np.std(prob_list)),
        "per_sample": per_sample,
    }
    # Free VRAM before next checkpoint
    del model
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_memorization_curve(results_by_epoch: dict, output_dir: str):
    """
    Plot RougeL recall and Probability vs. fine-tuning epochs.

    results_by_epoch: {epoch_int: result_dict}
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[warn] matplotlib not found — skipping plot")
        return

    epochs = sorted(results_by_epoch.keys())
    rl_means = [results_by_epoch[e]["rougeL_mean"] for e in epochs]
    rl_stds  = [results_by_epoch[e]["rougeL_std"]  for e in epochs]
    pr_means = [results_by_epoch[e]["probability_mean"] for e in epochs]
    pr_stds  = [results_by_epoch[e]["probability_std"]  for e in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Memorization Growth in a Discrete Diffusion LM (LLaDA-8B on TOFU forget10)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    colors = {"rougeL": "#d62728", "prob": "#1f77b4"}

    for ax, means, stds, label, color, ylabel in [
        (axes[0], rl_means,  rl_stds,  "RougeL Recall", colors["rougeL"], "RougeL Recall ↑ = more memorized"),
        (axes[1], pr_means,  pr_stds,  "Probability (ELBO proxy)", colors["prob"], "P(answer | question) ↑ = more memorized"),
    ]:
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        ax.plot(epochs, means_arr, marker="o", color=color, linewidth=2, markersize=6, label=label)
        ax.fill_between(epochs, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.15, color=color)
        ax.set_xlabel("Fine-tuning Epochs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "memorization_curve.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DLM memorization analysis (TOFU + LLaDA)")

    # Modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help="Evaluate all checkpoints under --checkpoint_dir")
    mode.add_argument("--plot_only", action="store_true",
                      help="Only re-plot from existing --results_json")

    # Paths
    p.add_argument("--model_path", default=None,
                   help="Path to a single checkpoint (single-checkpoint mode)")
    p.add_argument("--epoch", type=int, default=None,
                   help="Epoch number for single-checkpoint mode")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer path (defaults to --model_path or base model)")
    p.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Instruct",
                   help="Base model path / HF id for tokenizer fallback")
    p.add_argument("--checkpoint_dir", default="checkpoints/",
                   help="Dir with epoch-named subdirs for --sweep mode")
    p.add_argument("--checkpoint_pattern", default="ep{epoch}",
                   help="Subdirectory naming pattern (use {epoch})")
    p.add_argument("--epochs", nargs="+", type=int, default=None,
                   help="Epochs to evaluate in --sweep mode (default: auto-detect)")
    p.add_argument("--output_dir", default="outputs/", help="Where to save results")
    p.add_argument("--results_json", default=None,
                   help="Path to existing results JSON for --plot_only")

    # Eval hypers
    p.add_argument("--forget_split", default="forget10")
    p.add_argument("--n_samples", type=int, default=None,
                   help="How many forget QA pairs to evaluate (default: all)")
    p.add_argument("--steps", type=int, default=128,
                   help="Masked diffusion decoding steps")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--mask_samples", type=int, default=64,
                   help="Monte Carlo samples for Eq.(14) probability")
    p.add_argument("--mc_batch_size", type=int, default=32)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Plot only ─────────────────────────────────────────────────────────────
    if args.plot_only:
        results_json = args.results_json or os.path.join(args.output_dir, "memorization_results.json")
        with open(results_json) as f:
            data = json.load(f)
        results_by_epoch = {int(k): v for k, v in data["by_epoch"].items()}
        plot_memorization_curve(results_by_epoch, args.output_dir)
        return

    # ── Common eval kwargs ────────────────────────────────────────────────────
    eval_kwargs = dict(
        forget_split=args.forget_split,
        n_samples=args.n_samples,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        mask_samples=args.mask_samples,
        mc_batch_size=args.mc_batch_size,
    )

    # ── Single checkpoint ─────────────────────────────────────────────────────
    if not args.sweep:
        assert args.model_path, "--model_path required for single-checkpoint mode"
        assert args.epoch is not None, "--epoch required for single-checkpoint mode"
        tok = args.tokenizer or args.model_path
        result = evaluate_checkpoint(args.model_path, tok, **eval_kwargs)
        by_epoch = {args.epoch: result}

    # ── Sweep mode ────────────────────────────────────────────────────────────
    else:
        ckpt_dir = args.checkpoint_dir
        if args.epochs:
            epochs = args.epochs
        else:
            # Auto-detect: look for dirs matching ep<N>
            import re
            epochs = sorted(
                int(m.group(1))
                for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d))
                for m in [re.match(r"ep(\d+)$", d)]
                if m
            )
            if not epochs:
                raise ValueError(
                    f"No epoch directories found in {ckpt_dir}. "
                    "Use --epochs 10 100 200 ... or match pattern ep<N>."
                )
        print(f"Sweep epochs: {epochs}")

        by_epoch = {}
        for ep in epochs:
            subdir = args.checkpoint_pattern.format(epoch=ep)
            model_path = os.path.join(ckpt_dir, subdir)
            tok = args.tokenizer or model_path
            result = evaluate_checkpoint(model_path, tok, **eval_kwargs)
            by_epoch[ep] = result

            # Save incrementally so partial results are safe
            out = {"by_epoch": {str(k): v for k, v in by_epoch.items()}}
            with open(os.path.join(args.output_dir, "memorization_results.json"), "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    # ── Save full results ─────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "memorization_results.json")
    with open(out_path, "w") as f:
        json.dump({"by_epoch": {str(k): v for k, v in by_epoch.items()}}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>8}  {'RougeL':>8}  {'Probability':>12}")
    print("-" * 34)
    for ep in sorted(by_epoch):
        r = by_epoch[ep]
        print(f"{ep:>8}  {r['rougeL_mean']:>8.4f}  {r['probability_mean']:>12.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_memorization_curve({int(k): v for k, v in by_epoch.items()}, args.output_dir)


if __name__ == "__main__":
    main()
