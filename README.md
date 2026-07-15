# Memorization in Discrete Diffusion Language Models

> **Why does over-memorization make unlearning harder in DLMs?**
> This repository provides fine-tuning code, evaluation scripts, and a memorization analysis
> pipeline to study how training epochs affect memorization depth in LLaDA-8B — and why
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
| **Eq.(14) Probability** | ELBO lower bound on `P_θ(answer \| question)` — how confident the model is in the correct answer across all masking noise levels |

Both metrics increase with training epochs, reaching near-perfect recall and high
probability after ~1000 epochs on TOFU.

### Why over-memorization creates a robustness problem for unlearning

Standard unlearning methods (GA, GradDiff, ETW, DPO) modify model weights to reduce
`P_θ(answer | question)`. For a DLM, this corresponds to increasing loss at **masked
answer positions given only the question context** — i.e., the training objective
at `t→1` (fully masked answer).

However, a highly memorized DLM stores the factual association not just at `t=1`, but
across **all noise levels**:

```
t=1.0  (fully masked):   [Q] [MASK] [MASK] [MASK]      → model must recall from scratch
t=0.5  (half masked):    [Q] [answer_part1] [MASK]      → model conditions on partial answer
t=0.1  (barely masked):  [Q] [answer_almost] [MASK×1]   → trivial single-token fill-in
```

After deep memorization, the model can reconstruct the target answer from *any* partial
context — including from tokens it has already unmasked at earlier diffusion steps.
This is the **partial-context memorization pathway**:

```
Unlearning objective optimizes:   ↑ loss at t≈1 (fully masked)
But generation at inference uses: t: 1 → 0 (iterative unmasking)

→ Even if the model "forgets" at t=1, partially-unmasked intermediate states
  allow reconstruction of the forgotten answer.
```

### Formal statement

Let `x_A` = answer tokens, `x_Q` = question tokens. Generation computes:

```
P(x_A | x_Q) ≈ ∫₀¹ p_θ(x_A[masked_t] | x_Q, x_A[visible_t]) dt
```

Over-memorization embeds the factual association in **all** conditional distributions
`p_θ(· | x_Q, x_A[visible_t])`. Standard unlearning methods only reduce
`p_θ(· | x_Q, x_A[∅])` (the fully-masked case, `t=1`), leaving the partial-context
pathways intact.

**Training epoch count is therefore a meaningful control variable**: a model fine-tuned
for 10 epochs memorizes mainly at `t≈1`, while a model fine-tuned for 1000 epochs has
memorized across the full `t` range and resists unlearning regardless of method strength.

---

## Experimental Setup

- **Model**: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- **Dataset**: [TOFU](https://huggingface.co/datasets/locuslab/TOFU) `forget10` split — 200 fictitious author biographies (10% of the full 2000-QA dataset)
- **Fine-tuning**: Full fine-tuning on the `full` split (all 2000 QA pairs) for 10 to 1000 epochs, saving a checkpoint every 100 epochs
- **Training objective**: Masked diffusion loss (Eq. 14 from the LLaDA paper) — same objective used for both training and evaluation
- **Hyperparameters**: lr=2e-5, weight_decay=0.01, batch_size=4, grad_accum=4, warmup_epochs=1, max_length=512

---

## Repository Structure

```
mdu_unlearning_memorization_analysis/
├── analysis/
│   └── analyze_memorization.py   # Measure RougeL + Probability across checkpoints, plot curve
├── scripts/
│   └── finetune_tofu.py          # Standalone fine-tuning script (no external framework)
├── dllm/                         # DLM utilities: MDLMSampler for masked diffusion generation
├── pretrained/                   # LLaDA-8B-Instruct base model (downloaded by setup.sh)
├── checkpoints/                  # Fine-tuned epoch checkpoints (ep10/, ep100/, ...)
├── outputs/                      # Memorization results JSON + plots
├── requirements.txt
├── setup.sh                      # One-command environment setup
└── README.md
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/kyunyoo/mdu_unlearning_memorization_analysis.git
cd mdu_unlearning_memorization_analysis

# Installs dependencies, downloads LLaDA-8B-Instruct (~16 GB), and pre-caches TOFU
bash setup.sh
```

If you already have the model downloaded:
```bash
bash setup.sh --no-model
```

### 2. Fine-tune with epoch checkpoints

The following command replicates our experimental setup — fine-tuning LLaDA-8B on
TOFU `full` for 1000 epochs, saving a checkpoint every 100 epochs (plus epoch 10):

```bash
python scripts/finetune_tofu.py \
    --model_path pretrained/LLaDA-8B-Instruct \
    --output_dir checkpoints/ \
    --tofu_split full \
    --n_epochs 1000 \
    --save_every_n_epochs 100 \
    --extra_save_epochs 10 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05
```

Checkpoints are saved as `checkpoints/ep10/`, `checkpoints/ep100/`, ..., `checkpoints/ep1000/`.

**GPU requirement**: 1× A100 or H100 80 GB (LLaDA-8B in bfloat16).
Full 1000-epoch training takes approximately 20–22 hours on a single H100.

### 3. Measure memorization across checkpoints

```bash
# Evaluate all epoch checkpoints at once
python analysis/analyze_memorization.py --sweep \
    --checkpoint_dir checkpoints/ \
    --output_dir outputs/ \
    --forget_split forget10 \
    --steps 128 \
    --max_new_tokens 128 \
    --mask_samples 64

# Evaluate a single checkpoint
python analysis/analyze_memorization.py \
    --model_path checkpoints/ep100 \
    --epoch 100 \
    --output_dir outputs/ep100/ \
    --forget_split forget10

# Re-plot from saved results (no GPU needed)
python analysis/analyze_memorization.py --plot_only \
    --results_json outputs/memorization_results.json \
    --output_dir outputs/
```

Outputs:
- `outputs/memorization_results.json` — per-epoch metrics
- `outputs/memorization_curve.pdf` / `.png` — memorization growth plot

---

## Results

Measured on LLaDA-8B-Instruct fine-tuned on TOFU `full`, evaluated on the `forget10`
split (200 QA pairs):

| Epochs | RougeL Recall | Eq.(14) Probability |
|--------|:------------:|:-------------------:|
| 0 (base) | 0.148      | 0.078               |
| 10       | 0.316      | 0.225               |
| 1000     | 0.929      | 0.948               |

> The Eq.(14) Probability metric captures memorization across **all** noise levels `t ∈ [0,1]`,
> not just at `t=1`. Its rapid growth indicates that the model embeds the factual association
> across the full diffusion trajectory — the regime where standard unlearning methods fail.

---

## Connection to Machine Unlearning

This repo accompanies our paper studying machine unlearning for DLMs.
The 1000-epoch checkpoint serves as the "highly memorized" base model for unlearning experiments.
We apply standard methods (GA, GradDiff, ETW, DPO) and measure:

- **Forget quality**: RougeL + Probability on `forget10` (should drop to near-base-model level)
- **Retain quality**: RougeL + Probability on `retain90`, `world_facts`, `real_authors`

We find that standard methods struggle to reduce forget metrics without degrading retain
quality precisely because of the partial-context memorization pathways described above,
and propose improved objectives that target these pathways directly.

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
