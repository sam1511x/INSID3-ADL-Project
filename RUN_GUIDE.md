# Run guide: INSID3 reproduction, Task 1 and Task 2 (8 GB GPU)

This guide runs the three stages **in order** on a Linux PC with an NVIDIA GPU with 8 GB of VRAM:

1. **Reproduction** of INSID3 with the original scripts
2. **Task 1**: Adaptive Positional Debiasing
3. **Task 2**: Multi-Prototype Reference Matching

All commands run from the repository root. Every setting listed here was verified on Google Colab
(Tesla T4, PyTorch capped to **8 GB** with `--max-vram-gb 8`). Verified numbers are in
[results/COLAB_VERIFICATION.md](results/COLAB_VERIFICATION.md).

<!-- VERIFIED-SETTINGS -->

---

## 0. One-time setup

```bash
git clone https://github.com/sam1511x/INSID3-ADL-Project.git
cd INSID3-ADL-Project
bash scripts/0_setup.sh            # conda env "insid3" (python 3.10, torch cu126) + deps
conda activate insid3
```

### DINOv3 weights (one of the three options)

The code picks the first source that is available:

1. **Official `.pth` files** in `pretrain/` (Meta download links), e.g. `pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
2. **Official HuggingFace repos** (gated): accept the license on
   [dinov3-vitl16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and
   [dinov3-vitb16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then run `huggingface-cli login`.
3. **timm copy** of the same LVD-1689M weights (ungated, used automatically when 1 and 2 are not available).

To force a source, set `INSID3_ENCODER=hf` or `INSID3_ENCODER=timm`.

### Datasets

```bash
bash scripts/download_data.sh coco spair lung isic suim permis      # what was verified on Colab
bash scripts/download_data.sh isaid pascal_part lvis paco_part      # remaining benchmarks of the paper
```

See [docs/data.md](docs/data.md) for the expected folder layout. Disk use is about 20 GB for COCO,
40 GB for LVIS/PACO (COCO-2017 images), under 6 GB for each of the others; **PerMIS needs ~45 GB temporarily**
(it unpacks the TAO test videos before extracting its 216 episodes), so prepare it alone.

Optional CRF refinement (the paper's main Table 1 setting): `CRF=1 bash scripts/0_setup.sh`, then add `CRF=1` to every command below.

---

## 1. Reproduction

```bash
bash scripts/1_reproduction.sh                     # every dataset found in ./data, all folds
DATASETS="coco lung" bash scripts/1_reproduction.sh   # subset
```

This calls the **original** `inference_segmentation.py` (DINOv3-L, 1024 px, 1-shot) for each dataset and fold,
and `inference_keypoint_matching.py` twice on SPair-71k (original and debiased features, DINOv3-B, 768 px, k=20).
It finishes with `results/SUMMARY.md`.

Single run, the same as the upstream README:

```bash
python inference_segmentation.py --dataset coco --fold 0 --exp-name insid3-coco-f0
python inference_keypoint_matching.py --debiased --image-size 768 --model-size base --svd-comps 20
```

Paper reference numbers (mIoU, CRF on): COCO-20i 57.6 · LVIS-92i 41.8 · ISIC 54.4 · SUIM 54.9 · iSAID 52.1 ·
X-Ray 78.8 · PASCAL-Part 50.5 · PACO-Part 38.7 · PerMIS 67.0. SPair PCK@0.10, ViT-B: 46.8 (original) and 52.6 (debiased).

---

## 2. Task 1: Adaptive Positional Debiasing

```bash
bash scripts/2_task1.sh
```

This runs three steps:

| step | script | output |
|---|---|---|
| (a) analysis of the positional components | `experiments/task1_analyze_positional.py` | `output/t1-analysis-coco_*/positional_analysis.{png,json}` |
| (b) segmentation mIoU, all variants | `experiments/task1_adaptive_debias.py --dataset <ds> --fold <f>` | `output/t1-<ds>-f<f>_*/results.json` |
| (c) semantic correspondence PCK | `experiments/task1_adaptive_debias.py --task correspondence` | `output/t1-spair_*/results.json` |

Variants (`python experiments/task1_adaptive_debias.py --list-variants`):

| variant | mechanism |
|---|---|
| `hard (INSID3)` | original: `x - U_k U_kᵀ x` |
| `none` | no debiasing |
| `spectral rho=*` | **weighted projection**: `w_i = s_i² / (s_i² + rho·s_k²)` over all components |
| `soft_rank t=*` | **weighted projection** with a smooth rank cut-off: `w_i = sigmoid((k - i)/(t·k))` |
| `token eta=*` | **component-wise attenuation per token**: `alpha(p) = (‖U_kᵀx_p‖² / r_noise)^eta` |
| `template` | **image-dependent correction**: subtract only the fitted positional template `Σ β_i t_i(p) u_i` (β fitted per image) |
| `template_hybrid l=*` | template plus attenuation of the residual (l=0 gives template, l=1 gives hard) |

All variants are evaluated in **one pass on identical episodes**: features and target clustering are computed
once per episode and shared. Useful flags:

```bash
python experiments/task1_adaptive_debias.py --dataset coco --fold 0 \
    --variants "hard (INSID3)" "template" "d.mode=spectral,d.rho=3" \
    --max-episodes 300 --max-vram-gb 8 --exp-name t1-custom
```

---

## 3. Task 2: Multi-Prototype Reference Matching

```bash
bash scripts/3_task2.sh
```

`experiments/task2_multiproto.py` clusters the debiased reference-mask features into M prototypes
(spherical k-means, or agglomerative with `p.cluster=agglo`) and matches target regions against the set.
It runs M ∈ {2, 3, 4, 6, 8} × aggregation ∈ {max, mean (size-weighted), softmax, ot (many-to-many optimal transport)},
plus ablations (global prototype kept, clustering type, which stages use the set). `single (INSID3)` is the original.

```bash
python experiments/task2_multiproto.py --dataset pascal_part --fold 0 \
    --variants "single (INSID3)" "M=4 max" "p.num_protos=6,p.agg=softmax,p.temp=0.1"
```

---

## Results and tables

```bash
python experiments/summarize.py          # collects all output/*/ into results/SUMMARY.md
```

Each run folder holds `log.txt` and `results.json` (per-variant mIoU and FB-IoU, or PCK; episode count; peak VRAM; runtime).

## Troubleshooting

* **CUDA out of memory**: add `IMAGE_SIZE=768` (the README notes a minor drop), or close other GPU programs.
  The 8 GB cap only limits PyTorch's allocator, so leave about 0.5 GB for the CUDA context.
* **GPU without bf16** (pre-Ampere, e.g. GTX 10xx or RTX 20xx): it still works through autocast emulation, just slower.
* **Interrupted run**: partial `results.json` files are written every 50 episodes. Use `--max-episodes` for quick checks.
* `python experiments/check_equivalence.py` verifies that the extended model with default settings gives exactly the original INSID3 predictions.
