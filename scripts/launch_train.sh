#!/usr/bin/env bash
# Launch scripts/train.py under torchrun.
#
#   GPUS        comma-separated GPU indices (required), e.g. GPUS=0,1; GPUS=cpu runs on the CPU
#   NPROC       ranks (default: the number of GPUs listed; 1 with GPUS=cpu)
#   CONFIG      the YAML config (default: configs/sd3m_d2l.yaml)
#   OVERRIDE    space-separated dotted overrides, e.g. OVERRIDE="data.train_dir=/data/coco/train train.max_steps=200"
#               (no spaces inside one value: wandb.tags=[a,b])
#   RESUME_FROM a checkpoint (step_XXXXXXXX.pt) to continue the run from
#   MASTER_PORT torchrun's port (default: 29541)
#
# HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE default to 1 here; a non-empty value set in the environment is kept. The code loads the
# teacher from the local Hugging Face cache only (HF_HUB_CACHE), whatever these are: download it beforehand.
# Checkpoints go to <output_dir>/<name>/ (about 12 GB each; train.keep_last of them are kept).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${GPUS:-}" ]; then
    echo "[launch] set GPUS, e.g. GPUS=0,1 (or GPUS=cpu)" >&2
    exit 2
fi
DEVICE_ARGS=()
if [ "$GPUS" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    NPROC="${NPROC:-1}"
    DEVICE_ARGS=(--device cpu)
else
    export CUDA_VISIBLE_DEVICES="$GPUS"
    NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
fi
CONFIG="${CONFIG:-configs/sd3m_d2l.yaml}"
MASTER_PORT="${MASTER_PORT:-29541}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[launch] gpus=$GPUS nproc=$NPROC config=$CONFIG port=$MASTER_PORT HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[launch] override=${OVERRIDE:-<none>} resume_from=${RESUME_FROM:-<none>}"

ARGS=(--config "$CONFIG" "${DEVICE_ARGS[@]}")
if [ -n "${RESUME_FROM:-}" ]; then
    ARGS+=(--resume-from "$RESUME_FROM")
fi
OVERRIDES=()
if [ -n "${OVERRIDE:-}" ]; then
    read -r -a OVERRIDES <<<"$OVERRIDE"
fi

exec torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --master_port="$MASTER_PORT" \
    scripts/train.py "${ARGS[@]}" "${OVERRIDES[@]}"
