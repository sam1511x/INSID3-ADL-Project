#!/usr/bin/env bash
# Stage 2 - Task 1: Adaptive Positional Debiasing.
#   (a) analysis of the positional components, (b) segmentation mIoU sweep, (c) SPair PCK sweep.
#   DATASETS="coco lung" VARIANTS='"hard (INSID3)" "template"' bash scripts/2_task1.sh
set -e
source "$(dirname "$0")/common.sh"
DATASETS=${DATASETS:-"coco isic lung suim permis isaid pascal_part lvis paco_part"}
VAR_ARGS=""; [ -n "${VARIANTS:-}" ] && VAR_ARGS="--variants $VARIANTS"
COMMON="--model-size $MODEL_SIZE --image-size $IMAGE_SIZE --max-vram-gb $MAX_VRAM_GB $CRF_FLAG --output-dir $OUT"

# (a) positional-component analysis
if have_data coco; then
  python experiments/task1_analyze_positional.py --dataset coco --fold 0 --num-images 200 \
    $COMMON --exp-name t1-analysis-coco
fi
# (b) segmentation (all variants evaluated on identical episodes in one pass)
for ds in $DATASETS; do
  have_data "$ds" || continue
  for f in $(folds_of "$ds"); do
    eval python experiments/task1_adaptive_debias.py --dataset "$ds" --fold "$f" $COMMON \
      --exp-name "t1-$ds-f$f" $VAR_ARGS $EXTRA
  done
done
# (c) semantic correspondence on SPair-71k (README protocol)
if [ "${SKIP_SPAIR:-0}" != "1" ] && have_data spair; then
  eval python experiments/task1_adaptive_debias.py --task correspondence --model-size base --image-size 768 \
    --svd-comps 20 --max-vram-gb "$MAX_VRAM_GB" --output-dir "$OUT" --exp-name t1-spair $VAR_ARGS
fi
python experiments/summarize.py --output-dir "$OUT" --out results/SUMMARY.md
