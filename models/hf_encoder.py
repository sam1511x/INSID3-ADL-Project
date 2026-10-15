"""DINOv3 encoder loaded from the HuggingFace Hub (transformers format).

The official ``.pth`` checkpoints are distributed by Meta through signed links.
The same weights are also published (gated) on the HuggingFace Hub in
``transformers`` format. This wrapper exposes the subset of the original
DINOv3 API used by INSID3 (``get_intermediate_layers(x, n=1, reshape=True)``),
so the rest of the code base is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

HF_REPOS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class HFDinov3Encoder(nn.Module):
    """Thin wrapper around ``transformers.DINOv3ViTModel``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        cfg = model.config
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        # [CLS] + register tokens precede the patch tokens
        self.n_prefix_tokens = 1 + getattr(cfg, "num_register_tokens", 0)

    @classmethod
    def from_pretrained(cls, model_size: str = "large") -> "HFDinov3Encoder":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(HF_REPOS[model_size], dtype=torch.float32)
        return cls(model)

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1, reshape: bool = False,
                                norm: bool = True, **kwargs):
        """Return the (layer-normed) patch tokens of the last block, like DINOv3.

        Only ``n=1`` (last layer) is needed by INSID3.
        """
        if n != 1:
            raise NotImplementedError("HFDinov3Encoder only supports n=1 (last layer).")
        # last_hidden_state is already passed through the final LayerNorm
        out = self.model(pixel_values=x).last_hidden_state
        patches = out[:, self.n_prefix_tokens:]
        if reshape:
            B, _, H, W = x.shape
            h, w = H // self.patch_size, W // self.patch_size
            patches = patches.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return (patches,)


TIMM_NAMES = {
    "small": "vit_small_patch16_dinov3.lvd1689m",
    "base": "vit_base_patch16_dinov3.lvd1689m",
    "large": "vit_large_patch16_dinov3.lvd1689m",
}


class TimmDinov3Encoder(nn.Module):
    """Same DINOv3 LVD-1689M weights, as re-hosted (ungated) by timm on the HF Hub."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.patch_size = model.patch_embed.patch_size[0]
        self.embed_dim = model.embed_dim
        self.n_prefix_tokens = model.num_prefix_tokens

    @classmethod
    def from_pretrained(cls, model_size: str = "large") -> "TimmDinov3Encoder":
        import timm
        model = timm.create_model(TIMM_NAMES[model_size], pretrained=True, dynamic_img_size=True)
        return cls(model.eval())

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1, reshape: bool = False,
                                norm: bool = True, **kwargs):
        if n != 1:
            raise NotImplementedError("TimmDinov3Encoder only supports n=1 (last layer).")
        out = self.model.forward_features(x)  # final norm applied
        patches = out[:, self.n_prefix_tokens:]
        if reshape:
            B, _, H, W = x.shape
            h, w = H // self.patch_size, W // self.patch_size
            patches = patches.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return (patches,)


def load_hub_encoder(model_size: str = "large", source: str | None = None) -> nn.Module:
    """DINOv3 from the Hub: official gated transformers repo, else the ungated timm copy.

    ``source`` (or env ``INSID3_ENCODER``) forces 'hf' or 'timm'.
    """
    import os
    source = source or os.environ.get("INSID3_ENCODER")
    if source != "timm":
        try:
            enc = HFDinov3Encoder.from_pretrained(model_size)
            print(f"[INSID3] DINOv3-{model_size}: transformers weights {HF_REPOS[model_size]}")
            return enc
        except Exception as e:  # gated repo without access, offline, ...
            if source == "hf":
                raise
            print(f"[INSID3] HF repo unavailable ({type(e).__name__}); using timm weights")
    enc = TimmDinov3Encoder.from_pretrained(model_size)
    print(f"[INSID3] DINOv3-{model_size}: timm weights {TIMM_NAMES[model_size]}")
    return enc
