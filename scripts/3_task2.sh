#!/usr/bin/env bash
# Stage 3 - Task 2: Multi-Prototype Reference Matching (segmentation mIoU sweep).
#   DATASETS="coco pascal_part" VARIANTS='"single (INSID3)" "M=4 max"' bash scripts/3_task2.sh
set -e
source "$(dirname "$0")/common.sh"
DATASETS=${DATASETS:-"coco isic lung suim permis isaid pascal_part lvis paco_part"}
VAR_ARGS=""; [ -n "${VARIANTS:-}" ] && VAR_ARGS="--variants $VARIANTS"
COMMON="--model-size $MODEL_SIZE --image-size $IMAGE_SIZE --max-vram-gb $MAX_VRAM_GB $CRF_FLAG --output-dir $OUT"

for ds in $DATASETS; do
  have_data "$ds" || continue
  for f in $(folds_of "$ds"); do
    eval python experiments/task2_multiproto.py --dataset "$ds" --fold "$f" $COMMON \
      --exp-name "t2-$ds-f$f" $VAR_ARGS $EXTRA
  done
done
python experiments/summarize.py --output-dir "$OUT" --out results/SUMMARY.md
