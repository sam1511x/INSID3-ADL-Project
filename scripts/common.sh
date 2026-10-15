#!/usr/bin/env bash
# Shared settings for the run scripts. Override any of them from the environment, e.g.
#   IMAGE_SIZE=768 DATASETS="coco lung" bash scripts/2_task1.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL_SIZE=${MODEL_SIZE:-large}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
MAX_VRAM_GB=${MAX_VRAM_GB:-8}          # hard cap so runs behave like on an 8 GB GPU
CRF=${CRF:-0}                           # 1 -> --crf-mask-refinement (needs the CRF package)
OUT=${OUT:-output}
EXTRA=${EXTRA:-}                        # any extra flags, e.g. "--max-episodes 200"
CRF_FLAG=""; [ "$CRF" = "1" ] && CRF_FLAG="--crf-mask-refinement"

folds_of() {   # folds used by the paper for each benchmark
  case "$1" in
    coco|pascal_part|paco_part) echo "0 1 2 3" ;;
    isaid) echo "0 1 2" ;;
    lvis) echo "0 1 2 3 4 5 6 7 8 9" ;;
    *) echo "0" ;;
  esac
}

data_dir_of() {
  case "$1" in
    coco) echo data/COCO2014 ;; lvis) echo data/LVIS ;; pascal_part) echo data/Pascal-Part ;;
    paco_part) echo data/PACO-Part ;; isaid) echo data/iSAID ;; isic) echo data/ISIC ;;
    lung) echo data/LungSegmentation ;; suim) echo data/SUIM ;; permis) echo data/PerMIRS ;;
    spair) echo data/SPair-71k ;;
  esac
}

have_data() {
  if [ -d "$(data_dir_of "$1")" ]; then return 0; fi
  echo "!! skipping $1: $(data_dir_of "$1") not found (run: bash scripts/download_data.sh $1)"
  return 1
}
