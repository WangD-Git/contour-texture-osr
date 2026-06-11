"""Table IV baselines: torchvision ResNet-18 / MobileNetV3-S/L (1-ch stem, 10-class multilabel head)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, ResNet18_Weights

# Table IV parameter counts (1-ch stem + 10-class head)
PAPER_PARAM_COUNTS: dict[str, int] = {
    "resnet18": 11_175_370,
    "mobilenet_v3_small": 1_527_818,
    "mobilenet_v3_large": 4_214_554,
}


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    title: str
    ckpt_name: str
    paper_params: int
    lr: float = 5e-4
    lr_pretrained: float = 1e-4
    batch_size: int | None = None
    warmup_epochs: int = 5
    early_stop_patience: int = 30
    supports_pretrained: bool = True


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "resnet18",
        "ResNet-18",
        "baseline_resnet18_gc10_best.pth",
        paper_params=PAPER_PARAM_COUNTS["resnet18"],
        lr=3e-4,
        lr_pretrained=5e-5,
        batch_size=16,
        warmup_epochs=10,
        early_stop_patience=30,
    ),
    BaselineSpec(
        "mobilenet_v3_small",
        "MobileNetV3-Small",
        "baseline_mobilenet_v3_small_gc10_best.pth",
        paper_params=PAPER_PARAM_COUNTS["mobilenet_v3_small"],
        lr=5e-4,
        lr_pretrained=1e-4,
        batch_size=8,
        warmup_epochs=5,
        early_stop_patience=25,
    ),
    BaselineSpec(
        "mobilenet_v3_large",
        "MobileNetV3-Large",
        "baseline_mobilenet_v3_large_gc10_best.pth",
        paper_params=PAPER_PARAM_COUNTS["mobilenet_v3_large"],
        lr=3e-4,
        lr_pretrained=5e-5,
        batch_size=16,
        warmup_epochs=10,
        early_stop_patience=30,
    ),
)

BASELINE_BY_KEY: dict[str, BaselineSpec] = {s.key: s for s in BASELINE_SPECS}


class BaselineMultilabel(nn.Module):
    """Wrap torchvision backbone: input [B,1,H,W] → output {'logits': [B,K]}."""

    def __init__(self, backbone: nn.Module, *, arch_key: str):
        super().__init__()
        self.backbone = backbone
        self.arch_key = arch_key

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1] != 1:
            raise ValueError(f"baseline {self.arch_key} expects 1-channel input, got {x.shape[1]} channels")
        return {"logits": self.backbone(x)}


def _adapt_conv1_from_rgb(old: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _replace_first_conv(old: nn.Conv2d, *, pretrained: bool) -> nn.Conv2d:
    if pretrained and old.in_channels == 3:
        return _adapt_conv1_from_rgb(old)
    return nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )


def _patch_resnet_first_conv(net: models.ResNet, *, pretrained: bool) -> None:
    net.conv1 = _replace_first_conv(net.conv1, pretrained=pretrained)


def _patch_mobilenet_first_conv(net: models.MobileNetV3, *, pretrained: bool) -> None:
    """features[0] is Conv2dNormActivation; [0] within it is the first Conv2d."""
    stem_block = net.features[0]
    old_conv: nn.Conv2d = stem_block[0]
    stem_block[0] = _replace_first_conv(old_conv, pretrained=pretrained)


def _wrap(backbone: nn.Module, arch_key: str) -> BaselineMultilabel:
    return BaselineMultilabel(backbone, arch_key=arch_key)


def _replace_resnet_fc(net: models.ResNet, num_classes: int) -> None:
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)


def _replace_mobilenet_classifier(net: models.MobileNetV3, num_classes: int) -> None:
    last = net.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError(f"MobileNet final layer expected Linear, got {type(last)}")
    net.classifier[-1] = nn.Linear(last.in_features, num_classes)


def build_resnet18(num_classes: int = 10, *, pretrained: bool = False) -> BaselineMultilabel:
    if pretrained:
        # torchvision≥0.13: with weights, num_classes must be 1000 first, then swap to 10-class head
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        _replace_resnet_fc(net, num_classes)
        _patch_resnet_first_conv(net, pretrained=True)
    else:
        net = models.resnet18(weights=None, num_classes=num_classes)
        _patch_resnet_first_conv(net, pretrained=False)
    return _wrap(net, "resnet18")


def build_mobilenet_v3_small(num_classes: int = 10, *, pretrained: bool = False) -> BaselineMultilabel:
    if pretrained:
        net = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        _replace_mobilenet_classifier(net, num_classes)
        _patch_mobilenet_first_conv(net, pretrained=True)
    else:
        net = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        _patch_mobilenet_first_conv(net, pretrained=False)
    return _wrap(net, "mobilenet_v3_small")


def build_mobilenet_v3_large(num_classes: int = 10, *, pretrained: bool = False) -> BaselineMultilabel:
    if pretrained:
        net = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        _replace_mobilenet_classifier(net, num_classes)
        _patch_mobilenet_first_conv(net, pretrained=True)
    else:
        net = models.mobilenet_v3_large(weights=None, num_classes=num_classes)
        _patch_mobilenet_first_conv(net, pretrained=False)
    return _wrap(net, "mobilenet_v3_large")


_BUILDERS: dict[str, Callable[..., BaselineMultilabel]] = {
    "resnet18": build_resnet18,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "mobilenet_v3_large": build_mobilenet_v3_large,
}


def build_baseline(name: str, num_classes: int = 10, *, pretrained: bool = False) -> BaselineMultilabel:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown baseline {name!r}; options: {', '.join(sorted(_BUILDERS))}")
    spec = BASELINE_BY_KEY.get(key)
    if pretrained and spec is not None and not spec.supports_pretrained:
        raise ValueError(f"{key} does not support --pretrained")
    return _BUILDERS[key](num_classes=num_classes, pretrained=pretrained)


def resolve_ckpt_name(spec: BaselineSpec, *, pretrained: bool) -> str:
    if not pretrained:
        return spec.ckpt_name
    stem, ext = spec.ckpt_name.rsplit(".", 1)
    return f"{stem}_pt.{ext}"


def describe_architecture(model: BaselineMultilabel) -> str:
    bb = model.backbone
    if hasattr(bb, "conv1"):
        c = bb.conv1
        return (
            f"ResNet-18 | stem Conv2d(1→{c.out_channels}, k={c.kernel_size[0]}, s={c.stride[0]}) "
            f"| layer1-4 + avgpool | fc {bb.fc.in_features}→{bb.fc.out_features}"
        )
    if hasattr(bb, "features"):
        c = bb.features[0][0]
        lin0 = bb.classifier[0]
        last = bb.classifier[-1]
        return (
            f"MobileNetV3 | stem Conv2d(1→{c.out_channels}, k={c.kernel_size[0]}, s={c.stride[0]}) "
            f"| inverted residuals | cls {lin0.in_features}→{lin0.out_features}→{last.out_features}"
        )
    return type(bb).__name__


def verify_table45_models(num_classes: int = 10) -> list[str]:
    """Return warning list; empty means Table IV models pass structure/param check."""
    warnings: list[str] = []
    for key in ("resnet18", "mobilenet_v3_small", "mobilenet_v3_large"):
        spec = BASELINE_BY_KEY[key]
        m = build_baseline(key, num_classes=num_classes)
        from . import core

        n = core.count_parameters(m)
        if n != spec.paper_params:
            warnings.append(f"{key}: params={n} != paper {spec.paper_params} (Δ={n - spec.paper_params:+d})")
        out = m(torch.randn(2, 1, 192, 192))["logits"]
        if out.shape != (2, num_classes):
            warnings.append(f"{key}: bad output shape {tuple(out.shape)}")
    return warnings
