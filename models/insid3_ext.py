"""INSID3 extensions for the ADL project.

Task 1 - Adaptive Positional Debiasing (``DebiasConfig``):
    replaces the hard orthogonal-complement projection ``x - U_k U_k^T x`` by soft
    mechanisms that control how much positional information is removed.

Task 2 - Multi-Prototype Reference Matching (``ProtoConfig``):
    represents the annotated reference region with several prototypes (clusters of
    its debiased DINOv3 patch features) and matches target regions against the
    whole set with different aggregation strategies.

With the default configurations (``DebiasConfig(mode='hard')`` and
``ProtoConfig(num_protos=1)``) ``INSID3Ext`` produces exactly the same predictions
as the original ``INSID3`` model (see ``experiments/check_equivalence.py``).

``INSID3Ext`` additionally caches the encoder features and the target clustering
of the current episode, so that many variants can be evaluated on the same
episode at the cost of a single forward pass (see ``experiments/common.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import einops
import torch
import torch.nn.functional as F

from models.insid3 import INSID3
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask

DEBIAS_MODES = ("hard", "none", "spectral", "soft_rank", "token", "template", "template_hybrid")
PROTO_AGGS = ("mean", "max", "softmax", "ot")


# ──────── Configurations ────────

@dataclass
class DebiasConfig:
    """Positional debiasing configuration (Task 1).

    mode:
        hard            original INSID3: x - U_k U_k^T x
        none            no debiasing
        spectral        weighted projection, w_i = s_i^2 / (s_i^2 + rho * s_k^2) over all C components
        soft_rank       weighted projection, w_i = sigmoid((k - i) / (temp * k))  (smooth rank cut-off)
        token           component removal scaled per token by its positional energy:
                        alpha(p) = clamp(r(p) / r_noise, 0, 1) ** eta,  r(p) = ||U_k^T x(p)||^2
        template        image-dependent correction: removes only the part of each component that
                        matches its positional template (noise-image spatial pattern),
                        x - sum_i beta_i t_i(p) u_i  with beta_i fitted per image (least squares)
        template_hybrid template correction + attenuation of the residual along u_i by ``lam``
                        (lam=0 -> template, lam=1 -> hard)
    """
    mode: str = "hard"
    k: int | None = None      # number of positional components (None -> model.svd_components)
    rho: float = 1.0          # spectral
    temp: float = 0.1         # soft_rank (relative to k)
    eta: float = 1.0          # token
    beta_max: float = 1.5     # template / template_hybrid
    lam: float = 0.5          # template_hybrid

    def __post_init__(self):
        if self.mode not in DEBIAS_MODES:
            raise ValueError(f"Unknown debias mode {self.mode}; choose from {DEBIAS_MODES}")


@dataclass
class ProtoConfig:
    """Reference representation configuration (Task 2).

    num_protos:   number of reference prototypes (1 -> original single mean prototype)
    agg:          aggregation of similarities to the prototype set
                  mean    size-weighted average similarity
                  max     maximum similarity over prototypes
                  softmax softmax(sim / temp)-weighted similarity (smooth max)
                  ot      many-to-many: token level uses max, region level uses an entropic
                          optimal-transport plan between target regions and reference prototypes
    cluster:      'kmeans' (spherical k-means) or 'agglo' (agglomerative, average linkage)
    include_global: also keep the original mean prototype in the set
    min_tokens:   minimum number of reference tokens per prototype (limits num_protos on tiny masks)
    stages:       which matching stages use the prototype set; subset of 'fwd,seed,agg'
                  (other stages keep using the single mean prototype)
    """
    num_protos: int = 1
    agg: str = "max"
    cluster: str = "kmeans"
    include_global: bool = False
    min_tokens: int = 4
    temp: float = 0.05
    ot_eps: float = 0.05
    stages: str = "fwd,seed,agg"
    kmeans_iters: int = 30

    def __post_init__(self):
        if self.agg not in PROTO_AGGS:
            raise ValueError(f"Unknown aggregation {self.agg}; choose from {PROTO_AGGS}")

    @property
    def is_single(self) -> bool:
        return self.num_protos <= 1 and not self.include_global


# ──────── Prototype helpers ────────

def spherical_kmeans(X: torch.Tensor, k: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    """Spherical k-means with deterministic k-means++ init.

    Args:
        X: (N, C) L2-normalized features.
    Returns:
        (N,) labels in [0, k).
    """
    N = X.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = [X[int(torch.randint(N, (1,), generator=g))]]
    for _ in range(1, k):
        C = torch.stack(centers)
        d = (1 - X @ C.T).clamp_min(0).min(dim=1).values
        p = (d / d.sum()).cpu() if d.sum() > 0 else torch.full((N,), 1.0 / N)
        centers.append(X[int(torch.multinomial(p, 1, generator=g))])
    C = torch.stack(centers)
    labels = None
    for _ in range(iters):
        new_labels = (X @ C.T).argmax(dim=1)
        if labels is not None and torch.equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            idx = labels == j
            if idx.any():
                C[j] = F.normalize(X[idx].mean(0), dim=0)
    return labels


def sinkhorn(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float, iters: int = 100) -> torch.Tensor:
    """Entropic OT plan between histograms a (K,) and b (M,) for cost (K, M)."""
    Kmat = torch.exp(-cost / eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (Kmat @ v).clamp_min(1e-30)
        v = b / (Kmat.T @ u).clamp_min(1e-30)
    return u[:, None] * Kmat * v[None, :]


# ──────── Model ────────

class INSID3Ext(INSID3):
    """INSID3 with configurable positional debiasing and reference representation."""

    def __init__(self, *args, debias_cfg: DebiasConfig | None = None,
                 proto_cfg: ProtoConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.debias_cfg = debias_cfg or DebiasConfig()
        self.proto_cfg = proto_cfg or ProtoConfig()
        self._episode_cache = {}
        self.last_debias_stats = {}

    def set_variant(self, debias_cfg: DebiasConfig | None = None, proto_cfg: ProtoConfig | None = None):
        if debias_cfg is not None:
            self.debias_cfg = debias_cfg
        if proto_cfg is not None:
            self.proto_cfg = proto_cfg

    # ──────── Episode caching ────────

    def new_episode(self) -> None:
        """Drop cached features/clusterings (call once per new reference/target pair)."""
        self._episode_cache = {}

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        key = ("feat", tuple(imgs.shape))
        hit = self._episode_cache.get(key)
        if hit is not None and torch.equal(hit[0], imgs):
            return hit[1]
        fmaps = super()._extract_features(imgs)
        self._episode_cache[key] = (imgs, fmaps)
        return fmaps

    def _cluster_target(self, feat_tgt_flat: torch.Tensor) -> torch.Tensor:
        key = ("cluster", self.tau)
        if key not in self._episode_cache:
            self._episode_cache[key] = agglomerative_clustering(feat_tgt_flat, self.tau)
        return self._episode_cache[key]

    # ──────── Positional subspace (full spectrum) ────────

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Same estimate as INSID3, but keeps the full SVD for soft debiasing."""
        from torchvision.transforms.functional import normalize
        noise_img = normalize(
            torch.zeros(1, 3, self.image_size, self.image_size),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).to(device)
        noise_fmaps = self.encoder.to(device).get_intermediate_layers(
            noise_img, n=1, reshape=True
        )[0]
        noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)

        E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')
        E = E - E.mean(dim=1, keepdim=True)
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        self.U_full = U.contiguous()                         # (C, R)
        self.S_full = S.contiguous()                         # (R,)
        self.templates = (S[:, None] * Vh).contiguous()      # (R, hw): t_i = u_i^T E
        self.noise_grid = noise_fmaps.shape[-2:]
        # Positional energy ratio of noise tokens (reference for 'token' mode), per k
        coeffs = U.T @ F.normalize(einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)'), dim=0)
        self.noise_energy_cum = (coeffs ** 2).mean(dim=1).cumsum(0)  # (R,)
        return U[:, :self.svd_components].contiguous()

    def _component_weights(self, cfg: DebiasConfig, k: int, R: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(R, device=device, dtype=dtype)
        if cfg.mode == "hard":
            return (idx < k).to(dtype)
        if cfg.mode == "spectral":
            s2 = self.S_full.to(device=device, dtype=dtype) ** 2
            sk2 = s2[min(k, R) - 1]
            return s2 / (s2 + cfg.rho * sk2)
        if cfg.mode == "soft_rank":
            return torch.sigmoid((k - 0.5 - idx) / (cfg.temp * k))
        raise ValueError(cfg.mode)

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        cfg = self.debias_cfg
        if cfg.mode == "none":
            return fmaps_norm
        key = ("deb", repr(cfg), tuple(fmaps_norm.shape))
        hit = self._episode_cache.get(key)
        if hit is not None and torch.equal(hit[0], fmaps_norm):
            return hit[1]
        if cfg.mode == "hard" and (cfg.k is None or cfg.k == self.svd_components):
            out = super()._debias_features(fmaps_norm)
        else:
            out = self._soft_debias(fmaps_norm, cfg)
        self._episode_cache[key] = (fmaps_norm, out)
        return out

    def _soft_debias(self, fmaps_norm: torch.Tensor, cfg: DebiasConfig) -> torch.Tensor:

        B, T, C, H, W = fmaps_norm.shape
        in_dtype = fmaps_norm.dtype
        X = fmaps_norm.reshape(B * T, C, H * W).float()
        k = cfg.k if cfg.k is not None else self.svd_components
        U = self.U_full.to(X.device)
        R = U.shape[1]
        coeffs = torch.einsum('cr,ncp->nrp', U, X)  # (N, R, hw) projections u_i^T x(p)

        if cfg.mode in ("hard", "spectral", "soft_rank"):
            w = self._component_weights(cfg, k, R, X.device, X.dtype)
            removed = coeffs * w[None, :, None]
            self.last_debias_stats = {"removed_dims": float(w.sum())}

        elif cfg.mode == "token":
            ck = coeffs[:, :k]
            r = (ck ** 2).sum(dim=1)                                    # (N, hw)
            r_noise = self.noise_energy_cum[k - 1].to(X.device)
            alpha = (r / r_noise).clamp(0, 1) ** cfg.eta                # (N, hw)
            removed = torch.zeros_like(coeffs)
            removed[:, :k] = ck * alpha[:, None, :]
            self.last_debias_stats = {"alpha_mean": float(alpha.mean())}

        else:  # template / template_hybrid
            if (H, W) != tuple(self.noise_grid):
                raise ValueError("template debiasing requires the noise grid to match the feature grid")
            Tk = self.templates[:k].to(X.device)                        # (k, hw)
            ck = coeffs[:, :k]
            ck_c = ck - ck.mean(dim=2, keepdim=True)
            beta = (ck_c * Tk[None]).sum(-1) / (Tk ** 2).sum(-1).clamp_min(1e-12)[None]
            beta = beta.clamp(0.0, cfg.beta_max)                        # (N, k)
            pos = beta[:, :, None] * Tk[None]                           # fitted positional part
            if cfg.mode == "template_hybrid":
                pos = pos + cfg.lam * (ck - pos)
            removed = torch.zeros_like(coeffs)
            removed[:, :k] = pos
            self.last_debias_stats = {"beta_mean": float(beta.mean())}

        X_deb = X - torch.einsum('cr,nrp->ncp', U, removed)
        X_deb = X_deb.reshape(B, T, C, H, W).to(in_dtype)
        return F.normalize(X_deb, p=2, dim=2)

    # ──────── Reference prototypes ────────

    def _build_ref_prototypes(self, feat_refs_deb: torch.Tensor, ref_masks: torch.Tensor,
                              h: int, w: int):
        """Returns (global prototype (C,1), prototype set (M,C), weights (M,))."""
        cfg = self.proto_cfg
        S = feat_refs_deb.shape[1]
        per_shot_means, fg_all = [], []
        for s in range(S):
            mask_s = downsample_mask(ref_masks[s:s+1], h, w)
            fg = feat_refs_deb[0, s, :, mask_s]
            if fg.shape[1] > 0:
                per_shot_means.append(fg.mean(dim=1))
                fg_all.append(fg.T)
        ref_prototype = F.normalize(
            torch.stack(per_shot_means).mean(dim=0), p=2, dim=0
        ).unsqueeze(1)
        if cfg.is_single:
            return ref_prototype, ref_prototype.T, torch.ones(1, device=ref_prototype.device)

        X = F.normalize(torch.cat(fg_all, dim=0).float(), dim=1)  # (N, C)
        N = X.shape[0]
        M = max(1, min(cfg.num_protos, N // max(cfg.min_tokens, 1)))
        if M == 1:
            protos, weights = ref_prototype.T.float(), torch.ones(1, device=X.device)
        else:
            if cfg.cluster == "kmeans":
                labels = spherical_kmeans(X, M, iters=cfg.kmeans_iters)
            elif cfg.cluster == "agglo":
                from sklearn.cluster import AgglomerativeClustering
                D = (1 - (X @ X.T).clamp(-1, 1)).cpu().numpy()
                labels = torch.from_numpy(AgglomerativeClustering(
                    n_clusters=M, metric='precomputed', linkage='average').fit_predict(D)).to(X.device)
            else:
                raise ValueError(cfg.cluster)
            protos, weights = [], []
            for j in range(M):
                idx = labels == j
                if idx.any():
                    protos.append(F.normalize(X[idx].mean(0), dim=0))
                    weights.append(idx.sum().float())
            protos = torch.stack(protos)
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            if cfg.include_global:
                protos = torch.cat([ref_prototype.T.float(), protos], dim=0)
                weights = torch.cat([torch.ones(1, device=X.device), weights]) / 2.0
        return ref_prototype, protos.to(feat_refs_deb.dtype), weights.to(feat_refs_deb.dtype)

    def _aggregate(self, sims: torch.Tensor, weights: torch.Tensor, agg: str) -> torch.Tensor:
        """Aggregate (N, M) similarities to M prototypes into (N,)."""
        if sims.shape[1] == 1:
            return sims[:, 0]
        if agg == "mean":
            return sims @ weights
        if agg in ("max", "ot"):
            return sims.max(dim=1).values
        if agg == "softmax":
            a = torch.softmax(sims / self.proto_cfg.temp, dim=1)
            return (a * sims).sum(dim=1)
        raise ValueError(agg)

    def _token_sim(self, feat_tgt_deb: torch.Tensor, protos: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """(1,C,h,w) target features -> (h,w) aggregated similarity to the prototype set."""
        _, C, h, w = feat_tgt_deb.shape
        sims = feat_tgt_deb[0].reshape(C, -1).T @ protos.T  # (hw, M)
        return self._aggregate(sims, weights, self.proto_cfg.agg).reshape(h, w)

    def _region_sim_ot(self, cluster_protos: torch.Tensor, areas: torch.Tensor,
                       protos: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Many-to-many region score: expected similarity under an OT plan (K,)."""
        sims = cluster_protos.float() @ protos.float().T  # (K, M)
        a = areas.float() / areas.float().sum()
        plan = sinkhorn(1 - sims, a, weights.float(), self.proto_cfg.ot_eps)
        return ((plan * sims).sum(1) / plan.sum(1).clamp_min(1e-12)).to(cluster_protos.dtype)

    # ──────── Inference: segmentation ────────

    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        S = ref_images.shape[0]
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)

        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape

        ref_masks = ref_masks.unsqueeze(1)
        feat_tgt = fmaps_norm[:, S]

        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        ref_prototype, protos, weights = self._build_ref_prototypes(feat_refs_deb, ref_masks, h, w)
        stages = set(self.proto_cfg.stages.split(",")) if not self.proto_cfg.is_single else set()

        sim_maps = []
        for m in range(S):
            sim_maps.append(torch.einsum('bchw,bcxy->bhwxy', feat_refs_deb[:, m], feat_tgt_deb))
        candidate_mask = self._locate_candidates_ext(
            sim_maps, ref_masks, feat_tgt_deb, ref_prototype, protos, weights, "fwd" in stages, h, w
        )
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = self._cluster_target(feat_tgt_flat).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
        cluster_protos = compute_cluster_prototypes(feat_tgt_deb_flat, cluster_labels.view(-1), K)

        pred_mask = self._seed_and_aggregate_ext(
            candidate_mask, cluster_labels, cluster_protos, K,
            ref_prototype, protos, weights, stages, feat_tgt, feat_tgt_deb, h, w
        )
        return self._finalize_mask(pred_mask, tgt_image)

    def _locate_candidates_ext(self, sim_maps, ref_masks, feat_tgt_deb, ref_prototype,
                               protos, weights, use_set, h, w):
        if not use_set:
            return self._locate_candidates(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)
        # Forward: positive aggregated similarity to the prototype set
        sim_fwd = self._token_sim(feat_tgt_deb, protos, weights)
        forward_mask = sim_fwd > 0
        if forward_mask.sum() == 0:
            forward_mask = sim_fwd > float(torch.quantile(sim_fwd.float(), 0.9))
        # Backward: majority vote over nearest neighbours, identical to INSID3
        return forward_mask & self._backward_mask(sim_maps, ref_masks, h, w)

    def _backward_mask(self, sim_maps, ref_masks, h, w):
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]
            Hs, Ws = sim0.shape[:2]
            best_idx = sim0.permute(2, 3, 0, 1).reshape(h, w, -1).argmax(dim=2)
            rows, cols = best_idx // Ws, best_idx % Ws
            ref_mask_m = downsample_mask(ref_masks[m:m+1], Hs, Ws).squeeze(0)
            votes += ref_mask_m[rows, cols].to(torch.int32)
        return votes >= math.ceil(k / 2)

    def _seed_and_aggregate_ext(self, candidate_mask, cluster_labels, cluster_protos, K,
                                ref_prototype, protos, weights, stages, feat_tgt, feat_tgt_deb, h, w):
        if not stages:
            return self._seed_and_aggregate(candidate_mask, cluster_labels, cluster_protos, K,
                                            ref_prototype, feat_tgt, feat_tgt_deb, h, w)
        agg = self.proto_cfg.agg
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask

        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)
        all_ids, all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        areas_full = torch.zeros(K, device=cluster_labels.device)
        areas_full[all_ids] = all_areas.float()
        area_weights = per_cluster / all_areas

        ot_scores = None
        if agg == "ot" and ("seed" in stages or "agg" in stages):
            ot_scores = self._region_sim_ot(cluster_protos, areas_full, protos, weights)

        # Seed selection
        protos_matched = cluster_protos[matched_ids]
        if "seed" in stages:
            if ot_scores is not None:
                cross_sim_matched = ot_scores[matched_ids]
            else:
                cross_sim_matched = self._aggregate(protos_matched @ protos.T, weights, agg)
        else:
            cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
        seed_id = matched_ids[int(torch.argmax(cross_sim_matched).item())].item()

        # Intra-image similarity to seed (original feature space)
        feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
        orig_protos = compute_cluster_prototypes(feat_tgt_flat, cluster_labels.view(-1), K)
        intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

        # Cross-image similarity per cluster
        if "agg" in stages and ot_scores is not None:
            cross_sim = ot_scores
        else:
            if "agg" in stages:
                fg_sim = self._token_sim(feat_tgt_deb, protos, weights)
            else:
                fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
            cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
            for k in range(K):
                idx = (cluster_labels == k)
                cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        return final_mask


def build_insid3_ext_from_args(args, debias_cfg: DebiasConfig | None = None,
                               proto_cfg: ProtoConfig | None = None) -> INSID3Ext:
    from models import _build_encoder
    model = INSID3Ext(
        encoder=_build_encoder(args.model_size),
        image_size=args.image_size,
        svd_components=int(args.svd_comps),
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        crf_size=args.crf_size,
        resize_to_orig_size=False,
        device=args.device,
        debias_cfg=debias_cfg,
        proto_cfg=proto_cfg,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def cfg_to_dict(debias_cfg: DebiasConfig, proto_cfg: ProtoConfig) -> dict:
    return {"debias": asdict(debias_cfg), "proto": asdict(proto_cfg)}


__all__ = ["INSID3Ext", "DebiasConfig", "ProtoConfig", "build_insid3_ext_from_args",
           "cfg_to_dict", "replace"]
