#!/usr/bin/env bash
# Verification queue used on Google Colab (Tesla T4, PyTorch capped to 8 GB).
# Fixed episode subsets keep it within free-tier GPU time; every variant in a run
# sees identical episodes. Usage on the VM:  bash colab/verify_queue.sh [stage ...]
cd "$(dirname "$0")/.."
CAP="--max-vram-gb 8 --output-dir output"
T2="single (INSID3)|M=2 max|M=2 mean|M=2 softmax|M=2 ot|M=4 max|M=4 mean|M=4 softmax|M=4 ot|M=8 max|M=8 mean|M=8 softmax|M=8 ot|M=4 max +global|M=4 max agglo|M=4 max seed+agg only|M=4 max agg only"
IFS='|' read -r -a T2V <<< "$T2"

wait_for() { until [ -e "$1" ]; do echo "waiting for $1"; sleep 60; done; }

run() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }

stage_spair() {
  run python experiments/task1_adaptive_debias.py --task correspondence --model-size base --image-size 768 \
      --svd-comps 20 --max-episodes 2000 $CAP --exp-name colab-t1-spair
}
stage_lung() {
  run python experiments/task1_adaptive_debias.py --dataset lung --max-episodes 150 $CAP --exp-name colab-t1-lung
  run python experiments/task2_multiproto.py --dataset lung --max-episodes 150 $CAP --exp-name colab-t2-lung --variants "${T2V[@]}"
}
stage_coco() {
  wait_for data/COCO2014/splits
  run python experiments/task1_analyze_positional.py --dataset coco --fold 0 --num-images 100 $CAP --exp-name colab-t1-analysis-coco
  run python experiments/task1_adaptive_debias.py --dataset coco --fold 0 --max-episodes 250 $CAP --exp-name colab-t1-coco-f0
  run python experiments/task2_multiproto.py --dataset coco --fold 0 --max-episodes 250 $CAP --exp-name colab-t2-coco-f0 --variants "${T2V[@]}"
}
stage_isic() {
  wait_for data/ISIC/ISIC2018_Task1-2_Training_Input/3
  run python experiments/task1_adaptive_debias.py --dataset isic --max-episodes 150 $CAP --exp-name colab-t1-isic
  run python experiments/task2_multiproto.py --dataset isic --max-episodes 150 $CAP --exp-name colab-t2-isic --variants "${T2V[@]}"
}
stage_coco2() {   # follow-up: prototype set only in the aggregation stage, and Task1+Task2 combined
  run python experiments/task2_multiproto.py --dataset coco --fold 0 --max-episodes 250 $CAP --exp-name colab-t2b-coco-f0 \
      --variants "single (INSID3)" "M=2 max agg-only" "M=3 max agg-only" "M=4 max agg-only" "M=6 max agg-only" "M=8 max agg-only" \
                 "M=4 mean agg-only" "M=4 softmax agg-only" "M=4 ot agg-only" "M=8 softmax agg-only" "M=8 ot agg-only" \
                 "single + spectral" "M=4 max agg-only + spectral"
}
stage_lung2() {
  run python experiments/task2_multiproto.py --dataset lung --max-episodes 150 $CAP --exp-name colab-t2b-lung \
      --variants "single (INSID3)" "M=2 max agg-only" "M=4 max agg-only" "M=8 max agg-only" "M=4 mean agg-only" "M=8 mean agg-only" \
                 "M=4 ot agg-only" "M=8 ot agg-only" "single + spectral" "M=4 max agg-only + spectral"
}
stage_isic2() {
  run python experiments/task2_multiproto.py --dataset isic --max-episodes 150 $CAP --exp-name colab-t2b-isic \
      --variants "single (INSID3)" "M=2 max agg-only" "M=4 max agg-only" "M=8 max agg-only" "M=4 mean agg-only" \
                 "M=4 ot agg-only" "M=8 ot agg-only" "M=8 ot" "single + spectral" "M=4 max agg-only + spectral"
}
stage_repro() {   # original script, full benchmark
  run python inference_segmentation.py --dataset lung --exp-name colab-repro-lung
}

STAGES=${*:-"spair lung coco isic repro"}
for s in $STAGES; do stage_$s; done
python experiments/summarize.py --output-dir output --out results/COLAB_SUMMARY.md --include-partial
echo "=== QUEUE DONE $(date +%T)"
