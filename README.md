# Memorization in Discrete Diffusion Language Models

> **Why does over-memorization make unlearning harder in DLMs?**
> This repository provides training code, evaluation scripts, and a memorization analysis
> pipeline to study how fine-tuning epochs affect memorization depth in LLaDA — and why
> deeply memorized knowledge resists standard unlearning methods.

---

## The Core Problem: Why Highly Memorized = Hard to Unlearn

### Background: How DLMs generate answers

A Discrete Diffusion Language Model (DLM) like LLaDA learns a denoising distribution
`p_θ(x_i | x_masked)` — given a partially masked sequence, predict the missing tokens.

At inference, generation starts from a **fully masked** answer region and iteratively
unmasks tokens over T diffusion steps. Crucially, the model can use *any unmasked token
as conditioning context* — bidirectionally — which is fundamentally different from
autoregressive models.

### What "memorization" means for a DLM

We measure memorization via two proxies on the TOFU forget set:

| Metric | What it measures |
|---|---|
| **RougeL Recall** | How much of the ground-truth answer appears in the model's generation |
| **Eq.(14) Probability** | ELBO lower bound on `P_θ(answer \| question)` — how "confident" the model is in the correct answer at any masking noise level |

Both metrics increase monotonically with training epochs, reaching near-perfect recall
and high probability after ~500–1000 epochs on TOFU.

### Why over-memorization creates a robustness problem for unlearning

Standard unlearning methods (GA, GradDiff, ETW, DPO) modify model weights to reduce
`P_θ(answer | question)`. For a DLM, this corresponds to increasing loss at **masked
answer positions given the full question context (no answer tokens visible)** — i.e.,
the standard training objective at t→1 (fully masked).

However, a highly memorized DLM stores the factual association not just at `t=1`, but
across **all noise levels**:

```
t=1.0  (fully masked):   [Q] [MASK] [MASK] [MASK] → model must recall from scratch
t=0.5  (half masked):    [Q] [answer_part1] [MASK] → model conditions on partial answer
t=0.1  (barely masked):  [Q] [answer_almost] [MASK_1] → trivial single-token fill-in
```

After deep memorization, the model can reconstruct the target answer from *any* partial
context — including from other answer tokens it has already unmasked at earlier diffusion
steps. This is the **partial-context memorization pathway**:

```
Unlearning objective optimizes:  ↑ loss at t≈1 (fully masked)
But generation at inference uses: t: 1 → 0 (iterative unmasking)

→ Even if the model "forgets" at t=1, partially-unmasked states at t<1
  allow reconstruction of the forgotten answer.
```

### Formal statement

Let `x_A` = answer tokens, `x_Q` = question tokens. The generation process computes:

```
P(x_A | x_Q) ≈ ∫₀¹ p_θ(x_A[masked_t] | x_Q, x_A[visible_t]) dt
```

Over-memorization embeds the factual association in **all** conditional distributions
`p_θ(· | x_Q, x_A[visible_t])`. Standard unlearning methods only reduce
`p_θ(· | x_Q, x_A[∅])` (the fully-masked case, `t=1`), leaving the partial-context
pathways intact.

**This is why training epoch count is a meaningful variable**: a model fine-tuned for 10
epochs may only memorize at `t≈1`, while a model fine-tuned for 1000 epochs has
memorized across the full `t` range, making it resistant to unlearning regardless of
method.

### This repo: measuring memorization depth

We fine-tune LLaDA-8B-Instruct on TOFU (a synthetic biography dataset designed to
simulate real unlearning scenarios) for 10 to 1000 epochs, saving checkpoints every
100 epochs. We then measure:

1. **RougeL Recall** — generation-based (masked diffusion sampling)
2. **Eq.(14) Probability** — ELBO proxy at all noise levels (Monte Carlo average)

This lets us see *when* the model transitions from shallow to deep memorization, and
set up controlled experiments where the base model's memorization depth is an
experimental variable.

---

## Repository Structure

```
memorization-in-dlm/
├── analysis/
│   └── analyze_memorization.py   # Measure RougeL + Probability, plot curve
├── scripts/
│   └── finetune_tofu.py          # Standalone fine-tuning (no Hydra)
├── dllm/                         # DLM generation utilities (MDLMSampler)
├── pretrained/                   # Downloaded base model (after setup.sh)
├── checkpoints/                  # Fine-tuned checkpoints (ep10/, ep100/, ...)
├── outputs/                      # Memorization results + plots
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <this-repo-url>
cd memorization-in-dlm

# Install dependencies + download LLaDA-8B-Instruct + pre-cache TOFU
bash setup.sh
```

If you already have the model locally:
```bash
bash setup.sh --no-model
```

### 2. Fine-tune with epoch checkpoints

```bash
python scripts/finetune_tofu.py \
    --model_path pretrained/LLaDA-8B-Instruct \
    --output_dir checkpoints/ \
    --n_epochs 1000 \
    --save_every_n_epochs 100 \
    --extra_save_epochs 10 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-5
```

This saves checkpoints at `checkpoints/ep10/`, `checkpoints/ep100/`, ..., `checkpoints/ep1000/`.

**GPU requirement**: 1× A100/H100 80GB (LLaDA-8B in bfloat16). Training 1000 epochs
on TOFU (4000 samples) takes approximately 20–22 hours.

### 3. Analyze memorization

```bash
# Evaluate all checkpoints at once (runs sequentially)
python analysis/analyze_memorization.py --sweep \
    --checkpoint_dir checkpoints/ \
    --output_dir outputs/ \
    --steps 128 \
    --mask_samples 64

# Or evaluate a single checkpoint
python analysis/analyze_memorization.py \
    --model_path checkpoints/ep100 \
    --epoch 100 \
    --output_dir outputs/ep100/

# Re-plot from saved results (no model needed)
python analysis/analyze_memorization.py --plot_only \
    --results_json outputs/memorization_results.json \
    --output_dir outputs/
```

Outputs:
- `outputs/memorization_results.json` — all metrics by epoch
- `outputs/memorization_curve.pdf` / `.png` — publication-ready plot

---

## Expected Results

| Epochs | RougeL Recall | Probability |
|--------|--------------|-------------|
| 10     | ~0.05        | ~0.01       |
| 100    | ~0.20        | ~0.05       |
| 200    | ~0.45        | ~0.15       |
| 500    | ~0.75        | ~0.45       |
| 1000   | ~0.90        | ~0.70       |

The rapid growth of the **Probability** metric after ~200 epochs indicates that
memorization is embedding itself across the full noise schedule, not just at `t=1`.
This is the "highly memorized" regime where unlearning becomes significantly harder.

---

## Connection to Machine Unlearning

This repo accompanies our paper studying machine unlearning for DLMs.
The experimental setup uses the 1000-epoch checkpoint as the "highly memorized" base
model, then applies unlearning methods and measures:

- **Forget quality**: does the model stop generating the factual answer?
- **Retain quality**: does general capability (TOFU retain, world facts, real authors) stay intact?

We find that standard methods (GA, GradDiff, ETW, DPO) achieve different
trade-offs depending on memorization depth, and propose improved objectives that
target partial-context pathways directly.

---

## Citation

```bibtex
@article{todo,
  title   = {Unlearning in Discrete Diffusion Language Models},
  year    = {2025},
}
```

---

## License

MIT
