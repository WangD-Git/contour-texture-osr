"""Closed-set ↔ open-set bridge: 10-class weight loading and 9-class score adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .core import Config, apply_preset
from .model import build_model
from .runtime import resolve_cls_ckpt, torch_load_checkpoint

DEFAULT_CLOSED_CKPT = "checkpoints/ification_opt_8_16_32_dog_v2_best.pth"


def resolve_closed_ckpt(path: str | None = None) -> str:
    """Locate DoG v2 weights produced by closed-set train.py."""
    explicit = (path or "").strip()
    return resolve_cls_ckpt(explicit or DEFAULT_CLOSED_CKPT)


class ClosedCkptOSRAdapter(nn.Module):
    """10-class closed model → 9-class logit scoring adapter (--closed-ckpt eval)."""

    def __init__(self, model10: nn.Module, known_idx: list[int]):
        super().__init__()
        self.model = model10
        self.register_buffer(
            "known_idx",
            torch.tensor(known_idx, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x)
        idx = self.known_idx
        out["logits"] = out["logits"][:, idx]
        out["aux_contour"] = out["aux_contour"][:, idx]
        out["aux_texture"] = out["aux_texture"][:, idx]
        return out


def build_closed_model_for_osr_eval(
    cfg,
    known_idx: list[int],
    *,
    use_blur_pool: bool | None = None,
) -> ClosedCkptOSRAdapter:
    full = Config()
    apply_preset(full)
    if use_blur_pool is not None:
        full.use_blur_pool = bool(use_blur_pool)
    full.num_classes = 10
    full.data_root = cfg.data_root
    return ClosedCkptOSRAdapter(build_model(full), known_idx)


def load_closed_ckpt_into_osr(
    model: nn.Module,
    ckpt_path: str,
    known_idx: list[int],
    *,
    device: torch.device | str,
) -> list[str]:
    """Map a 10-class closed ckpt onto a 9-class OSR model (--init-closed warm start)."""
    state = torch_load_checkpoint(ckpt_path, device)
    sd10 = state.get("model_state_dict", state)
    sd9 = model.state_dict()
    missing: list[str] = []
    head_keys = (
        "head.weight",
        "head.bias",
        "aux_contour.weight",
        "aux_contour.bias",
        "aux_texture.weight",
        "aux_texture.bias",
    )

    for k, v9 in sd9.items():
        if k not in sd10:
            missing.append(k)
            continue
        v10 = sd10[k]
        if k in head_keys and v10.shape != v9.shape:
            sd9[k] = v10[known_idx]
        elif v10.shape == v9.shape:
            sd9[k] = v10
        else:
            missing.append(k)

    model.load_state_dict(sd9, strict=False)
    return missing
