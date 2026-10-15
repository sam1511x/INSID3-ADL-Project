#!/usr/bin/env bash
# Stage 3 - Task 2: Multi-Prototype Reference Matching (segmentation mIoU sweep).
#   DATASETS="coco pascal_part" VARIANTS='"single (INSID3)" "M=4 max"' bash scripts/3_task2.sh
set -e
source "$(dirname "$0")/common.sh"
DATASETS=${DATASETS:-"coco isic lung suim permis isaid pascal_part lvis paco_part"}
# Core variant set (verified on Colab). FULL=1 runs the whole registry (48 variants, ~2x slower).
CORE='"single (INSID3)" "M=2 max" "M=4 max" "M=8 max" "M=4 mean" "M=8 mean" "M=4 softmax" "M=4 ot" "M=8 ot" "M=4 max +global" "M=4 max agglo" "M=4 max seed+agg only" "M=2 max agg-only" "M=4 max agg-only" "M=8 max agg-only" "M=4 mean agg-only" "M=4 softmax agg-only" "M=4 ot agg-only" "M=8 ot agg-only" "single + spectral" "M=4 max agg-only + spectral"'
if [ -n "${VARIANTS:-}" ]; then VAR_ARGS="--variants $VARIANTS"
elif [ "${FULL:-0}" = "1" ]; then VAR_ARGS=""
else VAR_ARGS="--variants $CORE"; fi
COMMON="--model-size $MODEL_SIZE --image-size $IMAGE_SIZE --max-vram-gb $MAX_VRAM_GB $CRF_FLAG --output-dir $OUT"

for ds in $DATASETS; do
  have_data "$ds" || continue
  for f in $(folds_of "$ds"); do
    eval python experiments/task2_multiproto.py --dataset "$ds" --fold "$f" $COMMON \
      --exp-name "t2-$ds-f$f" $VAR_ARGS $EXTRA
  done
done
python experiments/summarize.py --output-dir "$OUT" --out results/SUMMARY.md
