"""Shared multilabel training loop: ASL model selection, warmup+cosine, AMP, early stop."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .core import Config, multilabel_val_metrics
from .runtime import RUN_DEVICE, amp_autocast


def amp_backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    max_grad_norm: float = 1.0,
) -> None:
    params = list(model.parameters())
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
        opt.step()


LossFn = Callable[[nn.Module, dict[str, torch.Tensor], torch.Tensor, int], torch.Tensor]
AfterLossFn = Callable[
    [nn.Module, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int],
    torch.Tensor,
]
OnBestFn = Callable[[float, int], None]
OnEpochEndFn = Callable[[int, dict[str, float], float, int], None]


def warmup_cosine_lambda(
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float,
    base_lr: float,
) -> Callable[[int], float]:
    min_ratio = max(0.0, eta_min / max(base_lr, 1e-12))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if total_epochs <= warmup_epochs:
            return min_ratio
        progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def train_multilabel_with_val(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    loss_fn: LossFn,
    *,
    lr: float | None = None,
    warmup_epochs: int = 0,
    after_loss_fn: AfterLossFn | None = None,
    extra_modules: list[nn.Module] | None = None,
    on_best: OnBestFn | None = None,
    on_epoch_end: OnEpochEndFn | None = None,
    verbose: bool = True,
) -> float:
    """Unified multilabel training: val τ=0.5 μF1 model selection + early stop."""
    train_lr = float(cfg.lr if lr is None else lr)
    opt = optim.AdamW(model.parameters(), lr=train_lr, weight_decay=cfg.weight_decay)
    warm = max(0, int(warmup_epochs))
    if warm > 0:
        sched: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=warmup_cosine_lambda(warm, cfg.epochs, cfg.cosine_eta_min, train_lr),
        )
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=max(0.0, cfg.cosine_eta_min)
        )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and RUN_DEVICE.type == "cuda")
    best_f1, best_state, best_epoch = 0.0, None, 0
    bad = 0
    taus = np.full((cfg.num_classes,), cfg.multilabel_threshold, dtype=np.float32)
    temps = np.ones((cfg.num_classes,), dtype=np.float32)
    patience = int(cfg.early_stop_patience)
    nb = RUN_DEVICE.type == "cuda" and not cfg.sync_h2d

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if extra_modules:
            for m in extra_modules:
                m.train()

        for x, y in train_loader:
            x, y = x.to(RUN_DEVICE, non_blocking=nb), y.to(RUN_DEVICE, non_blocking=nb)
            opt.zero_grad()
            with amp_autocast(cfg.use_amp):
                out = model(x)
                loss = loss_fn(model, out, y, epoch)
                if after_loss_fn is not None:
                    loss = after_loss_fn(model, out, y, loss, epoch)
            amp_backward_step(loss, model, opt, scaler, max_grad_norm=1.0)

        vm, temps, taus = multilabel_val_metrics(
            model,
            val_loader,
            cfg,
            taus=taus,
            temps=temps,
            epoch=epoch,
            use_tta=False,
        )
        sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        if vm["micro_f1"] > best_f1 + cfg.early_stop_min_delta:
            best_f1 = vm["micro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
            if on_best is not None:
                on_best(best_f1, best_epoch)
        else:
            bad += 1
        if on_epoch_end is not None:
            on_epoch_end(epoch, vm, best_f1, best_epoch)
        elif verbose and epoch % cfg.log_interval == 0:
            print(
                f"Ep{epoch:03d} | val_μF1={vm['micro_f1']:.4f} | best={best_f1:.4f} @ep{best_epoch} | lr={cur_lr:.2e}",
                flush=True,
            )
        if patience > 0 and bad >= patience:
            if verbose:
                print(f"early stop @ep{epoch} ({patience} epochs no gain, best@ep{best_epoch})", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    return float(best_f1)
