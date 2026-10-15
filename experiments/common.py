"""Shared utilities for the ADL project experiments.

All experiment scripts use the original INSID3 command-line options (``opts.py``)
plus a few extra flags, the original datasets (``datasets/``) and metrics
(``utils/metrics.py``). Several method variants are evaluated in a single pass:
for every episode the DINOv3 features and the target clustering are computed
once and shared by all variants, so each variant sees exactly the same episodes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from dataclasses import fields
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # weights (pretrain/) and data (data/) are resolved relative to the repo root

import opts  # noqa: E402
from datasets import build_dataset  # noqa: E402
from models.insid3_ext import DebiasConfig, ProtoConfig, build_insid3_ext_from_args, cfg_to_dict  # noqa: E402
from utils.metrics import AverageMeter, Evaluator, PCKEvaluator  # noqa: E402


# ──────── Arguments ────────

def get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description, parents=[opts.get_args_parser()])
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to evaluate (names from the script's registry or inline specs, "
                             "e.g. 'd.mode=template,d.lam=0.5' or 'p.num_protos=4,p.agg=max')")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Evaluate only the first N episodes (same episodes for every variant)")
    parser.add_argument("--max-vram-gb", type=float, default=None,
                        help="Cap PyTorch GPU memory (e.g. 8 to emulate an 8 GB GPU)")
    parser.add_argument("--task", default="segmentation", choices=["segmentation", "correspondence"])
    parser.add_argument("--list-variants", action="store_true", help="Print the variant registry and exit")
    return parser


def parse_variant(spec: str, registry: dict) -> tuple[str, DebiasConfig, ProtoConfig]:
    """Variant spec: registry name, or 'label=' optional prefix + comma separated d.<field>/p.<field>."""
    if spec in registry:
        d, p = registry[spec]
        return spec, d, p
    label = spec
    if "=" in spec.split(",")[0] and spec.split(",")[0].split("=")[0] == "label":
        first, _, spec = spec.partition(",")
        label = first.split("=", 1)[1]
    d_kwargs, p_kwargs = {}, {}
    d_types = {f.name: f.type for f in fields(DebiasConfig)}
    p_types = {f.name: f.type for f in fields(ProtoConfig)}
    for item in filter(None, spec.split(",")):
        if "=" not in item or "." not in item.split("=")[0]:
            raise ValueError(f"Unknown variant '{spec}'. Use a name from --list-variants "
                             f"or an inline spec like 'd.mode=template,p.num_protos=4'")
        key, val = item.split("=", 1)
        scope, name = key.split(".", 1)
        types, target = (d_types, d_kwargs) if scope == "d" else (p_types, p_kwargs)
        if name not in types:
            raise ValueError(f"Unknown field {key} in variant '{spec}'")
        target[name] = _cast(val, types[name])
    return label, DebiasConfig(**d_kwargs), ProtoConfig(**p_kwargs)


def _cast(val: str, typ):
    t = str(typ)
    if "bool" in t:
        return val.lower() in ("1", "true", "yes")
    if "int" in t and "float" not in t:
        return None if val.lower() == "none" else int(val)
    if "float" in t:
        return float(val)
    return val


def resolve_variants(args, registry: dict, default: list[str]):
    if args.list_variants:
        for name, (d, p) in registry.items():
            print(f"{name:28s} {d}  {p}")
        sys.exit(0)
    specs = args.variants or default
    return [parse_variant(s, registry) for s in specs]


# ──────── Setup ────────

def setup(args) -> str:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if args.max_vram_gb:
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            frac = min(1.0, args.max_vram_gb / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print(f"[VRAM] capped to {args.max_vram_gb:.1f} GB ({frac:.2f} of {total:.1f} GB)")
        torch.cuda.reset_peak_memory_stats()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    out_dir = join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, "log.txt"), "w") as fp:
        fp.write(" ".join(sys.argv) + "\n")
        fp.write(str(vars(args)) + "\n\n")
    return out_dir


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0


def save_results(out_dir: str, args, variants, metrics: dict, n_done: int, elapsed: float, final: bool):
    res = {
        "task": args.task,
        "dataset": args.dataset if args.task == "segmentation" else "spair",
        "fold": args.fold,
        "shots": args.shots,
        "model_size": args.model_size,
        "image_size": args.image_size,
        "svd_comps": args.svd_comps,
        "crf": bool(getattr(args, "crf_mask_refinement", False)),
        "episodes": n_done,
        "complete": final,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "command": " ".join(sys.argv),
        "variants": {
            label: {"config": cfg_to_dict(d, p), **metrics[label]} for label, d, p in variants
        },
    }
    with open(join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


# ──────── Segmentation sweep ────────

def run_segmentation(args, model, variants, out_dir: str) -> dict:
    ds = build_dataset(args.dataset, args=args)
    n = len(ds) if args.max_episodes is None else min(len(ds), args.max_episodes)
    meters = {label: AverageMeter(args.dataset, ds.class_ids, device=args.device) for label, _, _ in variants}
    start = time.time()

    def metrics():
        out = {}
        for label, meter in meters.items():
            miou, fbiou = meter.compute_iou()
            out[label] = {"mIoU": round(float(miou), 2), "FB-IoU": round(float(fbiou), 2)}
        return out

    pbar = tqdm(range(n), ncols=100)
    for idx in pbar:
        # Iterating the dataset in order with a fixed seed yields the same episodes
        # as the original DataLoader(shuffle=False, num_workers=0).
        batch = ds[idx]
        model.reset_state()
        for i in range(len(batch["ref_imgs"])):
            model.set_reference(batch["ref_imgs"][i], batch["ref_masks"][i])
        model.set_target(batch["tgt_img"])
        refs, masks, tgt = model._ref_images, model._ref_masks, model._tgt_image
        model.new_episode()

        tgt_mask = None
        for label, d, p in variants:
            model.set_variant(d, p)
            pred_mask = model.predict_mask(refs, masks, tgt)
            if tgt_mask is None:
                tgt_mask = F.interpolate(
                    batch["tgt_mask"].unsqueeze(0).unsqueeze(0).float(),
                    size=pred_mask.shape, mode="nearest",
                ).squeeze(0).squeeze(0) > 0.5
            inter, union = Evaluator.classify_prediction(
                pred_mask, tgt_mask, tgt_ignore_idx=batch.get("tgt_ignore_idx"))
            meters[label].update(inter, union, batch["class_id"].to(args.device))
        model.reset_state()

        if (idx + 1) % 50 == 0 or idx + 1 == n:
            m = metrics()
            first = variants[0][0]
            pbar.set_description(f"{first}: {m[first]['mIoU']:.1f}")
            save_results(out_dir, args, variants, m, idx + 1, time.time() - start, final=False)

    m = metrics()
    res = save_results(out_dir, args, variants, m, n, time.time() - start, final=True)
    report(out_dir, res, key="mIoU")
    return res


# ──────── Semantic correspondence sweep ────────

def run_correspondence(args, model, variants, out_dir: str) -> dict:
    ds = build_dataset("spair", args=args)
    n = len(ds) if args.max_episodes is None else min(len(ds), args.max_episodes)
    evals = {label: PCKEvaluator(pck_by="image") for label, _, _ in variants}
    start = time.time()

    def metrics():
        out = {}
        for label, ev in evals.items():
            pck = ev.get_result()
            out[label] = {f"PCK@{a:.2f}": round(float(v) * 100, 2) for a, v in zip(ev.alpha, pck)}
        return out

    pbar = tqdm(range(n), ncols=100)
    for idx in pbar:
        batch = ds[idx]
        model.reset_state()
        model.set_reference(batch["src_img"])
        model.set_target(batch["trg_img"])
        ref, tgt = model._ref_images, model._tgt_image
        src_kps = batch["src_kps"].to(args.device)
        trg_kps = batch["trg_kps"].to(args.device)
        model.new_episode()
        for label, d, p in variants:
            model.set_variant(d, p)
            pred = model.predict_keypoint(ref, src_kps, tgt, use_debiased=d.mode != "none")
            evals[label].calculate_pck(
                trg_kps.unsqueeze(0), pred.unsqueeze(0), torch.tensor([batch["n_pts"]]),
                [batch["category"]], torch.tensor([batch["pck_thresh"]]))
        model.reset_state()

        if (idx + 1) % 200 == 0 or idx + 1 == n:
            m = metrics()
            first = variants[0][0]
            pbar.set_description(f"{first}: {m[first]['PCK@0.10']:.1f}")
            save_results(out_dir, args, variants, m, idx + 1, time.time() - start, final=False)

    m = metrics()
    res = save_results(out_dir, args, variants, m, n, time.time() - start, final=True)
    report(out_dir, res, key="PCK@0.10")
    return res


def report(out_dir: str, res: dict, key: str):
    lines = [f"\n# {res['task']} | {res['dataset']} fold {res['fold']} | {res['shots']}-shot | "
             f"{res['model_size']} @ {res['image_size']} | episodes {res['episodes']} | "
             f"peak VRAM {res['peak_vram_gb']:.2f} GB | {res['elapsed_s']:.0f}s"]
    for label, v in res["variants"].items():
        vals = "  ".join(f"{k}={v[k]:.2f}" for k in v if k != "config")
        lines.append(f"{label:32s} {vals}")
    text = "\n".join(lines)
    print(text)
    with open(join(out_dir, "log.txt"), "a") as fp:
        fp.write(text + "\n")


def main(description: str, registry: dict, default_seg: list[str], default_corr: list[str] | None = None):
    parser = get_parser(description)
    args = parser.parse_args()
    variants = resolve_variants(args, registry, default_corr if args.task == "correspondence" and default_corr else default_seg)
    out_dir = setup(args)
    print(f"Variants: {[v[0] for v in variants]}")
    model = build_insid3_ext_from_args(args)
    model.to(args.device)
    model.eval()
    if args.task == "segmentation":
        return run_segmentation(args, model, variants, out_dir)
    return run_correspondence(args, model, variants, out_dir)
