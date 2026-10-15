"""Task 1 - Adaptive Positional Debiasing.

Compares the original hard orthogonal-complement projection of INSID3 with soft
debiasing mechanisms, on in-context segmentation (mIoU) and on semantic
correspondence (SPair-71k PCK). Same arguments as inference_segmentation.py /
inference_keypoint_matching.py, plus --task / --variants / --max-episodes.

Examples:
    # segmentation, COCO-20i fold 0 (all Task-1 variants in one pass)
    python experiments/task1_adaptive_debias.py --dataset coco --fold 0 --exp-name t1-coco-f0

    # semantic correspondence (README protocol: ViT-B, 768 px, 20 components)
    python experiments/task1_adaptive_debias.py --task correspondence --model-size base \
        --image-size 768 --svd-comps 20 --exp-name t1-spair

    python experiments/task1_adaptive_debias.py --list-variants
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import main  # noqa: E402
from models.insid3_ext import DebiasConfig as D, ProtoConfig as P  # noqa: E402

# k=None -> uses --svd-comps (500 for segmentation, 20 for the SPair protocol)
REGISTRY = {
    # references
    "hard (INSID3)":        (D("hard"), P()),
    "none":                 (D("none"), P()),
    # weighted projection: Wiener-like spectral shrinkage around the k-th singular value
    "spectral rho=0.5":     (D("spectral", rho=0.5), P()),
    "spectral rho=1":       (D("spectral", rho=1.0), P()),
    "spectral rho=2":       (D("spectral", rho=2.0), P()),
    # weighted projection: smooth rank cut-off
    "soft_rank t=0.05":     (D("soft_rank", temp=0.05), P()),
    "soft_rank t=0.1":      (D("soft_rank", temp=0.1), P()),
    "soft_rank t=0.2":      (D("soft_rank", temp=0.2), P()),
    # component-wise attenuation scaled per token by its positional energy
    "token eta=0.5":        (D("token", eta=0.5), P()),
    "token eta=1":          (D("token", eta=1.0), P()),
    "token eta=2":          (D("token", eta=2.0), P()),
    # image-dependent positional correction (template fitting)
    "template":             (D("template", beta_max=1.5), P()),
    "template_hybrid l=0.25": (D("template_hybrid", lam=0.25), P()),
    "template_hybrid l=0.5":  (D("template_hybrid", lam=0.5), P()),
    "template_hybrid l=0.75": (D("template_hybrid", lam=0.75), P()),
}

DEFAULT = list(REGISTRY)

if __name__ == "__main__":
    main("INSID3 Task 1: adaptive positional debiasing", REGISTRY, DEFAULT, DEFAULT)
