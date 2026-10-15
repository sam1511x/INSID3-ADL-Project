#!/usr/bin/env bash
# Stage 1 - reproduce INSID3 with the ORIGINAL scripts (inference_segmentation.py,
# inference_keypoint_matching.py). Usage:
#   bash scripts/1_reproduction.sh                       # all datasets present in ./data
#   DATASETS="coco lung" bash scripts/1_reproduction.sh
set -e
source "$(dirname "$0")/common.sh"
DATASETS=${DATASETS:-"coco isic lung suim permis isaid pascal_part lvis paco_part"}

# Original scripts do not cap VRAM; the setting (ViT-L @ 1024) fits in 8 GB.
for ds in $DATASETS; do
  have_data "$ds" || continue
  for f in $(folds_of "$ds"); do
    python inference_segmentation.py --dataset "$ds" --fold "$f" --model-size "$MODEL_SIZE" \
      --image-size "$IMAGE_SIZE" $CRF_FLAG --output-dir "$OUT" --exp-name "repro-$ds-f$f" $EXTRA
  done
done

# Semantic correspondence (README protocol: ViT-B, 768 px, 20 components)
if [ "${SKIP_SPAIR:-0}" != "1" ] && have_data spair; then
  python inference_keypoint_matching.py --model-size base --image-size 768 --svd-comps 20 \
    --output-dir "$OUT" --exp-name repro-spair-original
  python inference_keypoint_matching.py --debiased --model-size base --image-size 768 --svd-comps 20 \
    --output-dir "$OUT" --exp-name repro-spair-debiased
fi
python experiments/summarize.py --output-dir "$OUT" --out results/SUMMARY.md
