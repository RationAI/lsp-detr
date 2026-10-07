#!/bin/bash
# LSP-DETR-T detection arm: 2-class -> 3-class (non_tumor / tumor / Lymphocyte) fine-tune on the two
# annotated TILs-images patches. Same shape as scripts/dist_train_lsp.sh; the differences are the
# config, the `--tuning` checkpoint (the trained 2-class detector with a 3-row class head) and the
# audit's expected backbone LR.
#
# Usage:  bash det/scripts/dist_train_tils3.sh                          # 4 GPU, 80 ep, seed 0
#         NUM_GPUS=2 GPU_IDS=0,1 END_EPOCHS=2 bash det/scripts/dist_train_tils3.sh    # smoke
#         RESUME=<run>/last.pth bash det/scripts/dist_train_tils3.sh                  # resume

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DET_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${DET_DIR}"

MASTER_PORT=${MASTER_PORT:-7791}
NUM_GPUS=${NUM_GPUS:-4}                 # 4 divides the 8 val tiles exactly -> no DistributedSampler padding
GPU_IDS=${GPU_IDS:-"0,1,2,3"}
TRAIN_BATCH_SIZE_PER_GPU=${TRAIN_BATCH_SIZE_PER_GPU:-1}
TOTAL_TRAIN_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU))
TOTAL_VAL_BATCH_SIZE=${TOTAL_VAL_BATCH_SIZE:-${NUM_GPUS}}
CONFIG=${CONFIG:-"configs/LSP-T-tils3.yml"}
END_EPOCHS=${END_EPOCHS:-80}
ARM=${ARM:-strict-local}                # the arm checkpoint0022.pth was trained with (verified: AP 0.237 vs 0.091)
TUNING=${TUNING:-"${DET_DIR}/../ckpt/checkpoint0022_3class.pth"}
EXPECT_BACKBONE_LR=${EXPECT_BACKBONE_LR:-2.5e-6}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${DET_DIR}/output/tils3"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/[tils3]LSP-T_${ARM}_x${NUM_GPUS}_det-ft_${END_EPOCHS}ep"}
SEED=${SEED:-0}
PYTHON=${PYTHON:-python3}
EXTRA_UPDATES=${EXTRA_UPDATES:-""}
# NOTE: this machine exports a machine-wide `TF32=1`, so the knob is named TF32_POLICY
# (a plain `TF32` would be inherited as "1" and rejected by train.py --tf32).
TF32_POLICY=${TF32_POLICY:-keep}
RESUME=${RESUME:-""}

if ! command -v nvidia-smi >/dev/null 2>&1; then echo "Error: nvidia-smi not found." >&2; exit 1; fi
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( AVAILABLE_GPUS < NUM_GPUS )); then
    echo "Error: ${NUM_GPUS} GPUs required, only ${AVAILABLE_GPUS} detected." >&2; exit 1
fi
if [ -n "${RESUME}" ] && [ ! -f "${RESUME}" ]; then echo "Error: RESUME not found: ${RESUME}" >&2; exit 1; fi
if [ -z "${RESUME}" ] && [ ! -f "${TUNING}" ]; then echo "Error: TUNING checkpoint not found: ${TUNING}" >&2; exit 1; fi

echo "=========================================="
echo "LSP-DETR-det 3-class fine-tune (TILs-images)"
echo "=========================================="
echo "Arm: ${ARM}   GPUs: ${NUM_GPUS} (${GPU_IDS})   Train batch: ${TOTAL_TRAIN_BATCH_SIZE}   Val batch: ${TOTAL_VAL_BATCH_SIZE}"
echo "Config: ${CONFIG}"
echo "Tuning from: ${TUNING}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${END_EPOCHS}   Seed: ${SEED}   TF32: ${TF32_POLICY}"
echo "Extra updates: ${EXTRA_UPDATES}"
echo "=========================================="

export SAVE_INTERMEDIATE_VISUALIZE_RESULT=False
# This machine exports a malformed PYTORCH_CUDA_ALLOC_CONF (`key=value` instead of `key:value`),
# which SIGABRTs torch at CUDA init.
unset PYTORCH_CUDA_ALLOC_CONF
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL
export DOME_MATCH_THREADS=${DOME_MATCH_THREADS:-6}

mkdir -p "${OUTPUT_DIR}"

UPDATES=(epoches=${END_EPOCHS}
         train_dataloader.total_batch_size=${TOTAL_TRAIN_BATCH_SIZE}
         val_dataloader.total_batch_size=${TOTAL_VAL_BATCH_SIZE}
         LSPDetrDetection.center_mode=${ARM})
if [ -n "${EXTRA_UPDATES}" ]; then
    # shellcheck disable=SC2206
    UPDATES+=(${EXTRA_UPDATES})
fi

LOAD_ARGS=()
if [ -n "${RESUME}" ]; then
    LOAD_ARGS=(-r "${RESUME}"); echo "Resuming from: ${RESUME}"
else
    LOAD_ARGS=(-t "${TUNING}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
OMP_NUM_THREADS=1 \
"${PYTHON}" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    -c "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --tf32 "${TF32_POLICY}" \
    --expect-backbone-lr "${EXPECT_BACKBONE_LR}" \
    "${LOAD_ARGS[@]}" \
    -u "${UPDATES[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/train_run.log"

echo ""
echo "Training completed!"
