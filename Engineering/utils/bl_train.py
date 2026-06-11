"""GC10 multilabel CNN baseline training (warmup + cosine + early stop; shared train_loop)."""

from __future__ import annotations

import torch.nn as nn

from .core import AsymmetricLoss, Config
from .loop import train_multilabel_with_val


def train_baseline_multilabel(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    *,
    lr: float | None = None,
    warmup_epochs: int = 5,
    verbose: bool = True,
) -> float:
    """ASL single-head; ResNet / MobileNet Table IV baselines."""
    crit = AsymmetricLoss(cfg.asl_gamma_neg, cfg.asl_gamma_pos, cfg.asl_clip)

    def loss_fn(_m: nn.Module, out: dict, y, _epoch: int) -> torch.Tensor:
        return crit(out["logits"], y)

    return train_multilabel_with_val(
        model,
        train_loader,
        val_loader,
        cfg,
        loss_fn,
        lr=lr,
        warmup_epochs=warmup_epochs,
        verbose=verbose,
    )
