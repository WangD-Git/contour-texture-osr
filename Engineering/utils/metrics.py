"""GC10 multilabel CNN baseline evaluation (train.py protocol + stratified metrics)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from .core import (
    Config,
    eval_logits,
    logits_to_probs,
    multilabel_metrics,
    postprocess_multilabel_eval,
    predictions_from_probs,
)


def _metrics_at_fixed_threshold(logits: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, float]:
    temps = np.ones((cfg.num_classes,), dtype=np.float32)
    taus = np.full((cfg.num_classes,), cfg.multilabel_threshold, dtype=np.float32)
    probs = logits_to_probs(logits, temps)
    return multilabel_metrics(y, predictions_from_probs(probs, cfg, taus))


def eval_protocol_label(cfg: Config) -> str:
    """Paper primary-metric protocol label (matches cfg.use_tta_eval)."""
    return "T/τ+TTA" if cfg.use_tta_eval else "T/τ"


def evaluate_multilabel_report(
    model: nn.Module,
    val_loader,
    test_loader,
    cfg: Config,
) -> dict[str, Any]:
    """
    Closed-set evaluation shared by train.py / exp.py baseline:
    - plain: τ=0.5, no TTA
    - paper: fit T/τ on Val, apply on Test; TTA controlled by cfg.use_tta_eval (--no-tta disables)
    """
    v_plain, v_y = eval_logits(model, val_loader, cfg, use_tta=False)
    t_plain, t_y = eval_logits(model, test_loader, cfg, use_tta=False)
    plain_val = _metrics_at_fixed_threshold(v_plain, v_y, cfg)
    plain_test = _metrics_at_fixed_threshold(t_plain, t_y, cfg)

    do_tta = bool(cfg.use_tta_eval)
    v_logits, _ = eval_logits(model, val_loader, cfg, use_tta=do_tta)
    t_logits, _ = eval_logits(model, test_loader, cfg, use_tta=do_tta)
    _, _, cal_val, cal_test = postprocess_multilabel_eval(v_logits, v_y, t_logits, t_y, cfg)

    return {
        "plain_val": plain_val,
        "plain_test": plain_test,
        "cal_val": cal_val,
        "cal_test": cal_test,
        "use_tta": do_tta,
    }


def format_multilabel_eval_report(
    report: dict[str, Any],
    *,
    best_train_val: float,
    protocol: str | None = None,
) -> str:
    tag = protocol or ("T/τ+TTA" if report.get("use_tta", True) else "T/τ")
    pv, pt = report["plain_val"]["micro_f1"], report["plain_test"]["micro_f1"]
    cv, ct = report["cal_val"]["micro_f1"], report["cal_test"]["micro_f1"]
    lines = [
        f"  model selection best@τ=0.5 (no TTA) = {best_train_val:.4f}",
        f"  plain  Val/Test @τ=0.5 no TTA = {pv:.4f} / {pt:.4f}  (Δ={pv - pt:+.4f})",
        f"  paper  Val/Test @{tag:<9} = {cv:.4f} / {ct:.4f}  (Δ={cv - ct:+.4f})",
        (
            f"  T/τ gain over plain Val/Test = +{cv - pv:.4f} / +{ct - pt:.4f} "
            f"(per-class τ grid-searched on Val labels; paper-Val often exceeds Test)"
        ),
    ]
    return "\n".join(lines)
