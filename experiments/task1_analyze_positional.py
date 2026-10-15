"""Task 1 - analysis of the positional components identified by INSID3.

Measures how much each positional direction u_i (SVD of the noise-image features)
contributes to real image features, across images and across tokens:

* per-component energy  e_i = mean_p (u_i^T x_p)^2       (real images vs. noise image)
* per-component template fit R^2_i: how much of the component's spatial map in a real
  image is explained by its positional template (noise-image map)  -> "how positional"
* per-token positional energy ratio r(p) = ||U_k^T x_p||^2, separately for
  foreground (annotated object) and background tokens  -> semantic overlap
* fitted template amplitudes beta_i across images  -> image dependence
* average r(p) map over images  -> spatial (coordinate-aligned) structure

Example:
    python experiments/task1_analyze_positional.py --dataset coco --fold 0 --num-images 200 \
        --exp-name t1-analysis-coco
"""

import json
import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from experiments.common import get_parser, setup  # noqa: E402
from datasets import build_dataset  # noqa: E402
from models.insid3_ext import build_insid3_ext_from_args  # noqa: E402
from utils.data import downsample_mask, load_image  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


@torch.no_grad()
def analyze(args, model, out_dir):
    ds = build_dataset(args.dataset if args.task == "segmentation" else "spair", args=args)
    n = min(len(ds), args.num_images)
    U = model.U_full                                       # (C, R)
    R = U.shape[1]
    k = int(args.svd_comps)
    T = model.templates                                    # (R, hw)
    kmax = min(R, args.max_components)

    energy, r2, beta = [], [], []
    ratio_fg, ratio_bg, ratio_maps = [], [], []
    for idx in tqdm(range(n), ncols=100):
        batch = ds[idx]
        img = batch["tgt_img"] if "tgt_img" in batch else batch["src_img"]
        x, _ = load_image(img, model._transform, args.device)
        f = F.normalize(model._extract_features(x.unsqueeze(0)), dim=2)[0, 0]  # (C, h, w)
        C, h, w = f.shape
        X = f.reshape(C, -1).float()
        coeffs = U.T @ X                                   # (R, hw)
        energy.append((coeffs ** 2).mean(1).cpu())
        cc = coeffs[:kmax] - coeffs[:kmax].mean(1, keepdim=True)
        tt = T[:kmax]
        num = (cc * tt).sum(1)
        r2.append((num ** 2 / ((cc ** 2).sum(1) * (tt ** 2).sum(1)).clamp_min(1e-12)).cpu())
        beta.append((num / (tt ** 2).sum(1).clamp_min(1e-12)).cpu())
        rmap = (coeffs[:k] ** 2).sum(0).reshape(h, w)
        ratio_maps.append(rmap.cpu())
        if "tgt_mask" in batch:
            m = downsample_mask(batch["tgt_mask"][None, None].float().to(args.device) > 0, h, w)
            if m.any():
                ratio_fg.append(rmap[m].cpu())
            if (~m).any():
                ratio_bg.append(rmap[~m].cpu())
        model.new_episode()

    energy = torch.stack(energy)          # (n, R)
    r2 = torch.stack(r2)                  # (n, kmax)
    beta = torch.stack(beta)              # (n, kmax)
    ratio_maps = torch.stack(ratio_maps)  # (n, h, w)
    noise_energy = (model.S_full ** 2 / T.shape[1]).cpu()
    fg = torch.cat(ratio_fg) if ratio_fg else torch.empty(0)
    bg = torch.cat(ratio_bg) if ratio_bg else torch.empty(0)

    cum_img = energy.mean(0).cumsum(0)
    summary = {
        "num_images": n, "k": k, "feature_dim": int(U.shape[0]),
        "image_energy_in_top_k": float(cum_img[k - 1] / cum_img[-1]),
        "noise_energy_in_top_k": float(noise_energy[:k].sum() / noise_energy.sum()),
        "image_energy_in_top_k_per_image_std": float(
            (energy[:, :k].sum(1) / energy.sum(1)).std()),
        "template_r2_mean_first10": float(r2[:, :10].mean()),
        "template_r2_mean_first_k": float(r2[:, :min(k, kmax)].mean()),
        "template_r2_mean_tail": float(r2[:, min(k, kmax):].mean()) if kmax > k else None,
        "beta_mean_first10": float(beta[:, :10].mean()),
        "beta_std_across_images_first10": float(beta[:, :10].std(0).mean()),
        "token_ratio_fg_mean": float(fg.mean()) if fg.numel() else None,
        "token_ratio_bg_mean": float(bg.mean()) if bg.numel() else None,
        "token_ratio_mean": float(ratio_maps.mean()),
        "token_ratio_std_within_image": float(ratio_maps.flatten(1).std(1).mean()),
        "token_ratio_std_across_images": float(ratio_maps.flatten(1).mean(1).std()),
        "noise_token_ratio": float(model.noise_energy_cum[k - 1]),
    }
    with open(join(out_dir, "positional_analysis.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    torch.save({"energy": energy, "r2": r2, "beta": beta, "ratio_maps": ratio_maps,
                "noise_energy": noise_energy, "fg": fg, "bg": bg}, join(out_dir, "positional_analysis.pt"))
    print(json.dumps(summary, indent=2))
    plot(out_dir, energy, noise_energy, r2, beta, fg, bg, ratio_maps, k)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot(out_dir, energy, noise_energy, r2, beta, fg, bg, ratio_maps, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    idx = np.arange(1, energy.shape[1] + 1)

    ax = axes[0, 0]
    ax.semilogy(idx, noise_energy.numpy(), color=ORANGE, lw=2, label="noise image")
    m, s = energy.mean(0).numpy(), energy.std(0).numpy()
    ax.semilogy(idx, m, color=BLUE, lw=2, label="real images (mean)")
    ax.fill_between(idx, np.maximum(m - s, 1e-8), m + s, color=BLUE, alpha=0.15, linewidth=0)
    ax.axvline(k, color=MUTED, ls="--", lw=1)
    ax.text(k, ax.get_ylim()[1], f" k={k}", color=MUTED, va="top")
    ax.legend(frameon=False)
    _style(ax, "Energy per positional component", "component index i", "mean (u_i^T x)^2")

    ax = axes[0, 1]
    j = np.arange(1, r2.shape[1] + 1)
    ax.plot(j, r2.mean(0).numpy(), color=BLUE, lw=2, label="template R² (mean)")
    ax.plot(j, beta.mean(0).clamp(-1, 2).numpy(), color=AQUA, lw=2, label="fitted amplitude β (mean)")
    ax.axvline(k, color=MUTED, ls="--", lw=1)
    ax.legend(frameon=False)
    _style(ax, "How positional is each component in real images?", "component index i", "value")

    ax = axes[1, 0]
    if fg.numel() and bg.numel():
        hi = float(torch.quantile(torch.cat([fg, bg])[:200000], 0.995))
        bins = np.linspace(0, hi, 60)
        ax.hist(bg.numpy(), bins=bins, density=True, color=ORANGE, alpha=0.55, label="background tokens")
        ax.hist(fg.numpy(), bins=bins, density=True, color=BLUE, alpha=0.55, label="foreground tokens")
        ax.legend(frameon=False)
    else:
        ax.hist(ratio_maps.flatten().numpy(), bins=60, density=True, color=BLUE)
    _style(ax, "Per-token positional energy r(p) = ||U_k^T x_p||²", "r(p)", "density")

    ax = axes[1, 1]
    im = ax.imshow(ratio_maps.mean(0).numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Mean r(p) over images (spatial structure)", color=INK, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(join(out_dir, "positional_analysis.png"), dpi=150)

    # example per-token maps
    nshow = min(6, ratio_maps.shape[0])
    fig, axes = plt.subplots(1, nshow, figsize=(2.6 * nshow, 2.8))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(ratio_maps[i].numpy(), cmap="Blues")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image {i}", color=MUTED, fontsize=9)
    fig.suptitle("Per-token positional energy r(p), individual images", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(join(out_dir, "positional_token_maps.png"), dpi=150)
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    parser = get_parser("INSID3 Task 1: positional component analysis")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--max-components", type=int, default=1024)
    args = parser.parse_args()
    out_dir = setup(args)
    model = build_insid3_ext_from_args(args)
    model.to(args.device)
    model.eval()
    analyze(args, model, out_dir)
