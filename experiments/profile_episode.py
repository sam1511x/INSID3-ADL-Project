"""Profile one episode: encoder, clustering and per-variant cost; print the noise spectrum."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import get_parser, setup
from datasets import build_dataset
from models.insid3_ext import build_insid3_ext_from_args, DebiasConfig, ProtoConfig
from utils.clustering import agglomerative_clustering
import torch, torch.nn.functional as F

args = get_parser("profile").parse_args()
setup(args)
m = build_insid3_ext_from_args(args).to(args.device).eval()
S = m.S_full
print("noise singular values at", {i: round(float(S[i - 1]), 5) for i in (1, 10, 50, 100, 200, 300, 500, 700, 1000) if i <= len(S)})
for d in (DebiasConfig("spectral", rho=1.0), DebiasConfig("soft_rank", temp=0.1)):
    w = m._component_weights(d, m.svd_components, len(S), S.device, S.dtype)
    print(d.mode, "sum w =", round(float(w.sum()), 1), " w[0,99,499,699] =", [round(float(w[i]), 3) for i in (0, 99, 499, 699) if i < len(w)])
ds = build_dataset(args.dataset, args)
b = ds[0]
m.set_reference(b["ref_imgs"][0], b["ref_masks"][0]); m.set_target(b["tgt_img"])
refs, masks, tgt = m._ref_images, m._ref_masks, m._tgt_image
def t(fn):
    torch.cuda.synchronize(); s = time.time(); out = fn(); torch.cuda.synchronize(); return out, time.time() - s
m.new_episode()
fm, te = t(lambda: m._extract_features(torch.cat([refs, tgt]).unsqueeze(0)))
f = F.normalize(fm, dim=2)
_, tc = t(lambda: agglomerative_clustering(f[0, 1].reshape(f.shape[2], -1).T, m.tau))
print(f"encoder {te:.2f}s  clustering {tc:.2f}s")
m.new_episode(); m.predict_mask(refs, masks, tgt)  # fill cache
for name, d, p in [("hard", DebiasConfig(), ProtoConfig()), ("spectral", DebiasConfig("spectral"), ProtoConfig()),
                   ("template", DebiasConfig("template"), ProtoConfig()), ("M=4 max", DebiasConfig(), ProtoConfig(4, "max")),
                   ("M=4 ot", DebiasConfig(), ProtoConfig(4, "ot"))]:
    m.set_variant(d, p)
    pm, tv = t(lambda: m.predict_mask(refs, masks, tgt))
    deb = m._debias_features(f)
    m.set_variant(DebiasConfig(), ProtoConfig()); hard = m._debias_features(f)
    print(f"variant {name:10s} {tv:.2f}s fg={int(pm.sum())}  cos(deb, hard) mean={float((deb * hard).sum(2).mean()):.4f}")
