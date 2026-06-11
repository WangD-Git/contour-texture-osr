"""GC10 multilabel / NEU single-label training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .core import (
    AsymmetricLoss,
    Config,
    compose_dog_training_loss,
    decouple_loss,
    eval_logits,
    multilabel_metrics,
)
from .model import dual_dog_scale_balance_loss
from .runtime import RUN_DEVICE, amp_autocast
from .loop import amp_backward_step, train_multilabel_with_val


def onehot_to_class_index(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1).long()


def predict_single_label(logits: np.ndarray) -> np.ndarray:
    pred = np.zeros((logits.shape[0], logits.shape[1]), dtype=np.float32)
    pred[np.arange(logits.shape[0]), logits.argmax(axis=1)] = 1.0
    return pred


def single_label_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y_pred = predict_single_label(logits)
    acc = float((y_true == y_pred).all(axis=1).mean())
    ml = multilabel_metrics(y_true, y_pred)
    return {
        "accuracy": acc,
        "micro_f1": ml["micro_f1"],
        "micro_precision": ml["micro_precision"],
        "micro_recall": ml["micro_recall"],
    }


def evaluate_multilabel(model: nn.Module, val_loader, test_loader, cfg: Config):
    """Closed-set multilabel eval report (plain + paper); see metrics module."""
    from .metrics import evaluate_multilabel_report

    return evaluate_multilabel_report(model, val_loader, test_loader, cfg)


def evaluate_singlelabel(
    model: nn.Module, val_loader, test_loader, cfg: Config
) -> tuple[dict[str, float], dict[str, float]]:
    v_logits, v_y = eval_logits(model, val_loader, cfg, use_tta=False)
    t_logits, t_y = eval_logits(model, test_loader, cfg, use_tta=False)
    return single_label_metrics(v_y, v_logits), single_label_metrics(t_y, t_logits)


def train_multilabel_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    verbose: bool = True,
) -> float:
    """GC10: ASL + aux head (L_aux) + decouple (L_dec) + DoG scale balance."""
    crit = AsymmetricLoss(cfg.asl_gamma_neg, cfg.asl_gamma_pos, cfg.asl_clip)
    w_aux = float(np.clip(cfg.aux_contour_weight, 0.05, 0.95))
    w_bal = float(cfg.dog_scale_bal_weight)

    def loss_fn(m: nn.Module, out: dict, y: torch.Tensor, epoch: int) -> torch.Tensor:
        return compose_dog_training_loss(
            out,
            y,
            m,
            cfg,
            crit,
            epoch,
            w_aux=w_aux,
            w_bal=w_bal,
            device=RUN_DEVICE,
            dog_bal_fn=dual_dog_scale_balance_loss,
        )

    return train_multilabel_with_val(
        model,
        train_loader,
        val_loader,
        cfg,
        loss_fn,
        warmup_epochs=0,
        verbose=verbose,
    )


def train_singlelabel_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
) -> float:
    """NEU: CrossEntropy + light aux/decouple."""
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=max(0.0, cfg.cosine_eta_min))
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and RUN_DEVICE.type == "cuda")
    best_acc, best_state = 0.0, None
    bad = 0
    w_aux = float(np.clip(cfg.aux_contour_weight, 0.05, 0.95))
    warm_epochs = max(1, int(cfg.decouple_warmup_epochs))
    w_bal = float(cfg.dog_scale_bal_weight)

    for epoch in range(1, cfg.epochs + 1):
        warm = min(1.0, epoch / warm_epochs)
        model.train()
        for x, y in train_loader:
            nb = RUN_DEVICE.type == "cuda" and not cfg.sync_h2d
            x, y = x.to(RUN_DEVICE, non_blocking=nb), y.to(RUN_DEVICE, non_blocking=nb)
            idx = onehot_to_class_index(y)
            opt.zero_grad()
            with amp_autocast(cfg.use_amp):
                out = model(x)
                lm = crit(out["logits"], idx)
                la = w_aux * crit(out["aux_contour"], idx) + (1 - w_aux) * crit(out["aux_texture"], idx)
                ld = decouple_loss(out["feat_contour_l1"], out["feat_texture_l1"])
                loss = lm + cfg.aux_weight * warm * la + cfg.decouple_weight * warm * ld
                loss = loss + dual_dog_scale_balance_loss(model, w_bal, device=RUN_DEVICE)
            amp_backward_step(loss, model, opt, scaler, max_grad_norm=0.0)

        v_logits, v_y = eval_logits(model, val_loader, cfg, use_tta=False)
        vm = single_label_metrics(v_y, v_logits)
        sched.step()
        if vm["accuracy"] > best_acc + cfg.early_stop_min_delta:
            best_acc, best_state, bad = vm["accuracy"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if epoch % cfg.log_interval == 0:
            print(f"Ep{epoch:03d} | val_acc={vm['accuracy']:.4f} | best={best_acc:.4f}", flush=True)
        if cfg.early_stop_patience > 0 and bad >= cfg.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    return float(best_acc)
