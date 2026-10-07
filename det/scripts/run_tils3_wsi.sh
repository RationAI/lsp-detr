#!/bin/bash
# Batch whole-slide inference with the 3-class (non_tumor / tumor / Lymphocyte) LSP-DETR-det model.
#
# Writes, per slide, <slide file name incl. extension>_<TAG>.{zst,json,meta.json} into OUTPUT_DIR.
# The .zst carries the platform's 3-class viewer encoding (`tnt-lymph`), i.e. the same
# categoryId/termId/nt triples as the KongNet reference SSMH_BRS_HE_106.i2syntax_KongNet.zst:
#   non_tumor 1 / 66d54d2a89181badfeac2d75 / nt=true
#   tumor 2 / 66d54cd789181badfeac2d69 / nt=false
#   Lymphocyte 3 / 66d54cee89181badfeac2d6d / nt=true
# ('tnt-lymph-nt1': the KongNet/platform triples with the tumour/non-tumour numbers swapped, as the
#  delivered files use. ZST_CONVENTION=tnt-lymph gives the platform's own numbering instead.)
#
# Usage: bash det/scripts/run_tils3_wsi.sh [slide ...]      (no args -> every slide in WSI_DIR)
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DET_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${DET_DIR}"

WSI_DIR=${WSI_DIR:-/mnt/14T-2/tksong/tils-e2e-wsi}
RUN_DIR=${RUN_DIR:-"${DET_DIR}/output/tils3/[tils3]LSP-T_strict-local_x4_det-ft_80ep"}
# deployed 3-class checkpoint: the training run's best_stg1 if this clone has it, else the NAS copy
RESUME=${RESUME:-"${RUN_DIR}/best_stg1.pth"}
[ -f "${RESUME}" ] || RESUME=/mnt/nas2/LSP-DETR/TILS3/best_stg1.pth
CONFIG=${CONFIG:-configs/LSP-T-tils3.yml}
OUTPUT_DIR=${OUTPUT_DIR:-"${DET_DIR}/../results/tils3-wsi"}
TAG=${TAG:-LSPDETR-TNTL}
GPUS=${GPUS:-0,1,2,3,4,5}
BATCH_SIZE=${BATCH_SIZE:-4}
ZST_CONVENTION=${ZST_CONVENTION:-tnt-lymph-nt1}
CONF=${CONF:-0.5}          # the platform WSI recipe; our val sweep is flat 0.35-0.60 (macro-F1 0.792-0.796)
MAX_TILES=${MAX_TILES:-0}
PYTHON=${PYTHON:-python3}

# This machine's PYTORCH_CUDA_ALLOC_CONF is malformed and SIGABRTs torch at CUDA init.
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [ "$#" -gt 0 ]; then
    SLIDES=("$@")
else
    # .i2syntax goes through the Philips pixel engine (TILs-Inference/data/philips_pixel_engine).
    # The bindings are ABI-tagged per interpreter; if `import philips_pixel_engine` warns that the
    # bindings are missing, build one for this Python (see TILS3_FINETUNE.md §7).
    mapfile -t SLIDES < <(find "${WSI_DIR}" -maxdepth 1 -type f \
        \( -name '*.tif' -o -name '*.svs' -o -name '*.mrxs' -o -name '*.ndpi' -o -name '*.i2syntax' -o -name '*.isyntax' \) | sort)
fi

mkdir -p "${OUTPUT_DIR}"
echo "=========================================="
echo "LSP-DETR-det 3-class WSI inference"
echo "  checkpoint : ${RESUME}"
echo "  config     : ${CONFIG}"
echo "  slides     : ${#SLIDES[@]}"
echo "  output     : ${OUTPUT_DIR}   tag=${TAG}"
echo "  gpus       : ${GPUS}   batch=${BATCH_SIZE}   conf=${CONF}   zst=${ZST_CONVENTION}"
echo "=========================================="

ok=0; fail=0
for slide in "${SLIDES[@]}"; do
    base="$(basename "${slide}")"
    out="${OUTPUT_DIR}/${base}_${TAG}.zst"
    if [ -f "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] ${base} (exists; FORCE=1 to redo)"; ok=$((ok+1)); continue
    fi
    echo ""; echo "######## ${base}  ($(date '+%H:%M:%S'))"
    extra=()
    [ "${MAX_TILES}" != "0" ] && extra+=(--max-tiles "${MAX_TILES}")
    if "${PYTHON}" -u wsi_infer.py -i "${slide}" -c "${CONFIG}" -r "${RESUME}" --weights ema \
         -o "${OUTPUT_DIR}" --tag "${TAG}" --gpus "${GPUS}" --batch-size "${BATCH_SIZE}" \
         --conf-threshold "${CONF}" --zst-convention "${ZST_CONVENTION}" "${extra[@]}" \
         2>&1 | tee "${OUTPUT_DIR}/${base}_${TAG}.log"; then
        ok=$((ok+1))
    else
        echo "[FAIL] ${base}"; fail=$((fail+1))
    fi
done

echo ""; echo "=========================================="
echo "done: ${ok} ok, ${fail} failed  ->  ${OUTPUT_DIR}"
