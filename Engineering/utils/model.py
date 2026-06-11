"""Learnable DoG contour-texture frontend + five-stage dual-stream backbone (v2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    GAUSSIAN_BLUR_KERNEL_SIZE,
    GAUSSIAN_BLUR_SIGMA,
    GRAY_NORM_EPS,
    Config,
    _gaussian_kernel,
    build_dual_stream_net,
)

DOG_KERNEL_SIZE = 7
DOG_SCALES: tuple[tuple[float, float], ...] = (
    (0.5, 0.8),
    (0.8, 1.3),
    (1.0, 1.6),
)


def _gaussian_2d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = (size - 1) / 2.0
    ax = torch.arange(size, device=device, dtype=dtype) - r
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return g / g.sum()


def _dog_kernel(size: int, sigma_c: float, sigma_s: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g_c = _gaussian_2d(size, sigma_c, device, dtype)
    g_s = _gaussian_2d(size, sigma_s, device, dtype)
    k = g_c - g_s
    return k - k.mean()


def _normalize01_per_image(t: torch.Tensor, eps: float = GRAY_NORM_EPS) -> torch.Tensor:
    lo = t.amin(dim=(-2, -1), keepdim=True)
    hi = t.amax(dim=(-2, -1), keepdim=True)
    return (t - lo) / (hi - lo + eps)


class LearnableDoGContourPrep(nn.Module):
    """Grayscale → [learnable DoG contour, fixed Gaussian-difference texture]."""

    def __init__(self, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        pad = DOG_KERNEL_SIZE // 2
        self.dog_convs = nn.ModuleList(
            [nn.Conv2d(1, 1, DOG_KERNEL_SIZE, padding=pad, bias=False) for _ in range(n_scales)]
        )
        mid = 8
        self.mixer = nn.Sequential(
            nn.Conv2d(n_scales + 1, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=False),
            nn.Conv2d(mid, n_scales, 1, bias=True),
        )
        self.last_scale_w: torch.Tensor | None = None
        self._init_dog_weights()

    def _init_dog_weights(self) -> None:
        with torch.no_grad():
            for i, conv in enumerate(self.dog_convs):
                sc = DOG_SCALES[i % len(DOG_SCALES)]
                k = _dog_kernel(
                    DOG_KERNEL_SIZE,
                    sc[0],
                    sc[1],
                    conv.weight.device,
                    conv.weight.dtype,
                )
                conv.weight.copy_(k.view(1, 1, DOG_KERNEL_SIZE, DOG_KERNEL_SIZE))

    def _fixed_texture(self, x: torch.Tensor) -> torch.Tensor:
        pad = GAUSSIAN_BLUR_KERNEL_SIZE // 2
        g = _gaussian_kernel(
            GAUSSIAN_BLUR_KERNEL_SIZE,
            GAUSSIAN_BLUR_SIGMA,
            x.device,
            x.dtype,
        )
        blurred = F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), g)
        return _normalize01_per_image((x - blurred).abs())

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        if gray.dim() == 3:
            gray = gray.unsqueeze(0)
        if gray.size(1) != 1:
            gray = gray[:, :1]
        dogs = [_normalize01_per_image(conv(gray)) for conv in self.dog_convs]
        stack = torch.cat(dogs, dim=1)
        w = F.softmax(self.mixer(torch.cat([gray, stack], dim=1)), dim=1)
        self.last_scale_w = w
        contour = (w * stack).sum(dim=1, keepdim=True)
        texture = self._fixed_texture(gray)
        return torch.cat([contour, texture], dim=1)


class GrayRepeatPrep(nn.Module):
    """Ablation w/o DoG: pass grayscale through dual stream as duplicated 2-ch input."""

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        if gray.dim() == 3:
            gray = gray.unsqueeze(0)
        if gray.size(1) != 1:
            gray = gray[:, :1]
        return torch.cat([gray, gray], dim=1)


def build_input_prep(cfg: Config) -> nn.Module:
    if getattr(cfg, "use_dog_prep", True):
        return LearnableDoGContourPrep()
    return GrayRepeatPrep()


def dual_dog_scale_balance_loss(
    model: nn.Module,
    weight: float = 0.001,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    prep = getattr(model, "prep", None)
    if prep is None or not isinstance(prep, LearnableDoGContourPrep):
        dev = device if device is not None else "cpu"
        return torch.tensor(0.0, device=dev)
    w = getattr(prep, "last_scale_w", None)
    if w is None:
        dev = device if device is not None else "cpu"
        return torch.tensor(0.0, device=dev)
    mean_w = w.mean(dim=(0, 2, 3))
    target = 1.0 / float(w.size(1))
    return float(weight) * ((mean_w - target).pow(2).sum())


class DualStreamDoGNet(nn.Module):
    """Optional DoG frontend → five-stage asymmetric dual stream (contour 5×5 / texture 3×3)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.use_dog_prep = bool(getattr(cfg, "use_dog_prep", True))
        self.prep = build_input_prep(cfg)
        self.backbone = build_dual_stream_net(
            cfg,
            contour_kernel=5,
            texture_kernel=3,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.size(1) != 1:
            x = x[:, :1]
        return self.backbone(self.prep(x))


def build_model(cfg: Config) -> DualStreamDoGNet:
    return DualStreamDoGNet(cfg)
