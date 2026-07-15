#!/usr/bin/env bash
# setup.sh — one-time setup for the memorization-in-dlm repo
# Run once before training or analysis:
#   bash setup.sh [--model-only] [--no-model]
#
# What it does:
#   1. Install Python dependencies
#   2. Download LLaDA-8B-Instruct (requires ~16 GB disk, ~5 min on fast connection)
#   3. Pre-cache the TOFU dataset from Hugging Face

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${REPO_ROOT}/pretrained/LLaDA-8B-Instruct"
SKIP_MODEL=false
MODEL_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-model)   SKIP_MODEL=true  ;;
    --model-only) MODEL_ONLY=true  ;;
  esac
done

echo "=================================================="
echo " memorization-in-dlm setup"
echo "=================================================="

# ── 1. Python dependencies ──────────────────────────────────────────────────
if [ "$MODEL_ONLY" = false ]; then
  echo ""
  echo "[1/3] Installing Python dependencies..."
  pip install -r "${REPO_ROOT}/requirements.txt" --quiet
  echo "  Done."
fi

# ── 2. Download LLaDA-8B-Instruct ──────────────────────────────────────────
if [ "$SKIP_MODEL" = false ]; then
  echo ""
  echo "[2/3] Downloading LLaDA-8B-Instruct (~16 GB)..."
  echo "      Saving to: ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"

  python - <<PYEOF
import os
from huggingface_hub import snapshot_download

local_dir = "${MODEL_DIR}"
print(f"  Downloading GSAI-ML/LLaDA-8B-Instruct → {local_dir}")
snapshot_download(
    repo_id="GSAI-ML/LLaDA-8B-Instruct",
    local_dir=local_dir,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print("  Model download complete.")
PYEOF
else
  echo "[2/3] Skipping model download (--no-model)"
fi

# ── 3. Pre-cache TOFU dataset ───────────────────────────────────────────────
if [ "$MODEL_ONLY" = false ]; then
  echo ""
  echo "[3/3] Pre-caching TOFU dataset from Hugging Face..."
  python - <<PYEOF
from datasets import load_dataset
for split in ["full", "forget10", "retain90", "world_facts", "real_authors", "forget10_perturbed"]:
    try:
        ds = load_dataset("locuslab/TOFU", split, split="train")
        print(f"  {split}: {len(ds)} samples cached")
    except Exception as e:
        print(f"  {split}: skip ({e})")
PYEOF
  echo "  Done."
fi

echo ""
echo "=================================================="
echo " Setup complete!"
echo ""
echo " Model:   ${MODEL_DIR}"
echo ""
echo " Quick start:"
echo "   # Fine-tune (10 → 1000 epochs, save every 100):"
echo "   python scripts/finetune_tofu.py \\"
echo "     --model_path ${MODEL_DIR} \\"
echo "     --output_dir checkpoints/ \\"
echo "     --n_epochs 1000 --save_every_n_epochs 100"
echo ""
echo "   # Analyze memorization across checkpoints:"
echo "   python analysis/analyze_memorization.py --sweep \\"
echo "     --checkpoint_dir checkpoints/ \\"
echo "     --output_dir outputs/"
echo "=================================================="
