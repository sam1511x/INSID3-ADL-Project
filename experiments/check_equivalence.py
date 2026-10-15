"""Sanity check: INSID3Ext with default configs == original INSID3, and every variant runs.

    python experiments/check_equivalence.py                     # real DINOv3 (GPU)
    python experiments/check_equivalence.py --random-tiny --device cpu   # random tiny ViT (logic test only)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.insid3 import INSID3  # noqa: E402
from models.insid3_ext import INSID3Ext, DebiasConfig, ProtoConfig  # noqa: E402
from models.hf_encoder import HFDinov3Encoder  # noqa: E402


def tiny_encoder():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=128, patch_size=16, num_register_tokens=4)
    return HFDinov3Encoder(DINOv3ViTModel(cfg).eval())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-size", default="large")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--svd-comps", type=int, default=500)
    a = ap.parse_args()

    if a.random_tiny:
        enc, a.image_size, a.svd_comps = tiny_encoder(), 256, 20
    else:
        from models import _build_encoder
        enc = _build_encoder(a.model_size)
    kw = dict(image_size=a.image_size, svd_components=a.svd_comps, device=a.device, resize_to_orig_size=True)
    if a.random_tiny:
        kw.update(tau=0.6, merge_threshold=0.0)
    orig = INSID3(encoder=enc, **kw).to(a.device).eval()
    ext = INSID3Ext(encoder=enc, **kw).to(a.device).eval()

    ref, ref_m, tgt = "assets/ref_cat_image.jpg", "assets/ref_cat_mask.png", "assets/target_cat_image.jpg"

    def run(model):
        model.reset_state()
        model.set_reference(ref, ref_m)
        model.set_target(tgt)
        if isinstance(model, INSID3Ext):
            model.new_episode()
        return model.segment()

    p0 = run(orig)
    p1 = run(ext)
    print(f"default variant identical to INSID3: {torch.equal(p0, p1)}  (fg pixels {int(p0.sum())})")
    assert torch.equal(p0, p1)

    # hard projection through the generic soft path must match the closed form
    f = torch.nn.functional.normalize(torch.randn(1, 2, enc.embed_dim, 4, 4, device=a.device), dim=2)
    ext.set_variant(DebiasConfig("hard"))
    ref_out = ext._debias_features(f)
    gen_out = ext._soft_debias(f, DebiasConfig("hard", k=a.svd_comps))
    err = (ref_out - gen_out).abs().max().item()
    print(f"generic hard projection max abs err: {err:.2e}")
    assert err < 1e-4
    lam1 = ext._soft_debias(f, DebiasConfig("template_hybrid", lam=1.0)) if f.shape[-2:] == ext.noise_grid else None
    del lam1

    # template_hybrid with lam=1 must equal hard (needs full-size features)
    ext.new_episode()
    ext.set_variant(DebiasConfig("hard"))
    x, _ = __import__("utils.data", fromlist=["load_image"]).load_image(tgt, ext._transform, a.device)
    fm = torch.nn.functional.normalize(ext._extract_features(x.unsqueeze(0)), dim=2)
    e1 = (ext._soft_debias(fm, DebiasConfig("template_hybrid", lam=1.0)) - ext._debias_features(fm)).abs().max().item()
    print(f"template_hybrid(lam=1) vs hard max abs err: {e1:.2e}")
    assert e1 < 1e-4

    # every Task-1/Task-2 variant runs
    from experiments.task1_adaptive_debias import REGISTRY as R1
    from experiments.task2_multiproto import REGISTRY as R2
    ext.reset_state()
    ext.set_reference(ref, ref_m)
    ext.set_target(tgt)
    refs, masks, t = ext._ref_images, ext._ref_masks, ext._tgt_image
    ext.new_episode()
    for name, (d, p) in {**R1, **R2}.items():
        ext.set_variant(d, p)
        pm = ext.predict_mask(refs, masks, t)
        print(f"  {name:28s} fg={int(pm.sum()):8d}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("OK")


if __name__ == "__main__":
    main()
