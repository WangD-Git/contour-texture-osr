"""ResNet-18 + MSP open-set baseline (Table VI; same leave-one-unknown protocol as open.py)."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import bl_model, bl_train, core, open_train
from .pipeline import checkpoint_dir, prepare_model_on_device
from .runtime import RUN_DEVICE, amp_autocast, seed_status_line, torch_load_checkpoint

RESNET_SPEC = bl_model.BASELINE_BY_KEY["resnet18"]

# Table VI rows → unknown folder ID (matches open_train.OSR_EXPERIMENTS)
TABLE46_EXPERIMENTS: dict[str, int] = dict(open_train.OSR_EXPERIMENTS)

# Primary metric per experiment for end-of-run comparison with our method
TABLE46_OURS_METRIC: dict[str, tuple[str, float]] = {
    "A": ("main-head entropy", 0.9423),
    "B": ("JS disagreement", 0.9240),
    "C": ("main-head entropy", 0.4001),
    "D": ("main-head entropy", 0.4512),
}


def osr_ckpt_name(unknown_folder_id: int, *, pretrained: bool = False) -> str:
    tag = "_pretrained" if pretrained else ""
    return f"baseline_osr_resnet18_unknown{int(unknown_folder_id)}{tag}_best.pth"


def _configure_resnet_osr_cfg(
    *,
    unknown_folder_id: int,
    data_root: str | None,
    batch_size: int | None,
    epochs: int | None,
    seed: int,
    num_workers: int | None = None,
) -> open_train.OSRConfig:
    """ResNet+MSP OSR cfg: 9-class head, closed-split seed aligned with DoG OSR."""
    cfg = open_train.OSRConfig()
    cfg.apply_defaults()
    cfg.unknown_folder_id = int(unknown_folder_id)
    cfg.apply_osr_model_dims()
    cfg.seed = int(seed)
    cfg.use_channels_last = False
    cfg.early_stop_patience = RESNET_SPEC.early_stop_patience
    if data_root:
        cfg.data_root = data_root.strip()
    if batch_size is not None:
        cfg.batch_size = max(1, int(batch_size))
    elif RESNET_SPEC.batch_size is not None:
        cfg.batch_size = RESNET_SPEC.batch_size
    if epochs is not None:
        cfg.epochs = max(1, int(epochs))
    if num_workers is not None:
        cfg.num_workers = max(0, int(num_workers))
    return cfg


def resolve_osr_ckpt(path: str, unknown_folder_id: int, *, pretrained: bool = False) -> str:
    p = path.strip()
    if p:
        if os.path.isfile(p):
            return os.path.abspath(p)
        for cand in (p, os.path.join("checkpoints", p)):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        raise FileNotFoundError(f"checkpoint not found: {p}")
    default = str(checkpoint_dir() / osr_ckpt_name(unknown_folder_id, pretrained=pretrained))
    if not os.path.isfile(default):
        raise FileNotFoundError(f"checkpoint not found: {default}")
    return default


@torch.no_grad()
def collect_msp_scores(model: nn.Module, loader, cfg: core.Config) -> np.ndarray:
    """MSP score: 1 − max_k σ(logit_k)."""
    model.eval()
    chunks: list[np.ndarray] = []
    for x, _ in loader:
        x = x.to(RUN_DEVICE, non_blocking=True)
        with amp_autocast(cfg.use_amp):
            out = model(x)
            pm = torch.sigmoid(out["logits"].float())
        chunks.append((1.0 - pm.max(dim=1).values).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


@torch.no_grad()
def evaluate_resnet_msp_osr(
    model: nn.Module,
    closed_test_loader,
    unknown_loader,
    val_loader,
    cfg: core.Config,
) -> dict[str, float]:
    cal_logits, cal_y = core.eval_logits(model, val_loader, cfg, use_tta=True)
    te_logits, te_y = core.eval_logits(model, closed_test_loader, cfg, use_tta=True)
    _, _, _, closed_m = core.postprocess_multilabel_eval(cal_logits, cal_y, te_logits, te_y, cfg)

    s_closed = collect_msp_scores(model, closed_test_loader, cfg)
    s_unknown = collect_msp_scores(model, unknown_loader, cfg)
    y_bin = np.concatenate(
        [np.zeros(len(s_closed), dtype=np.float32), np.ones(len(s_unknown), dtype=np.float32)]
    )
    scores = np.concatenate([s_closed, s_unknown])

    return {
        "closed_micro_f1": float(closed_m["micro_f1"]),
        "auroc_msp": open_train.binary_auroc(y_bin, scores),
        "msp_closed_mean": float(s_closed.mean()),
        "msp_unknown_mean": float(s_unknown.mean()),
        "msp_gap_mean": float(s_unknown.mean() - s_closed.mean()),
        "n_closed_eval": float(len(s_closed)),
        "n_unknown_eval": float(len(s_unknown)),
    }


def save_osr_checkpoint(
    path: str,
    *,
    model: nn.Module,
    cfg: open_train.OSRConfig,
    best_f1: float,
    metrics: dict[str, float],
    pretrained: bool,
) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "best_val_micro_f1": best_f1,
        "config": asdict(cfg),
        "arch": "baseline_resnet18_osr_msp",
        "pretrained": bool(pretrained),
        "known_class_indices": cfg.known_class_indices,
        "unknown_folder_id": cfg.unknown_folder_id,
        "osr_metrics": metrics,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def run_resnet18_osr_once(
    *,
    unknown_folder_id: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    data_root: str | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    eval_only: bool = False,
    ckpt: str = "",
    pretrained: bool = False,
    seed: int = 44,
    deterministic: bool = False,
    num_workers: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = _configure_resnet_osr_cfg(
        unknown_folder_id=unknown_folder_id,
        data_root=data_root,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        num_workers=num_workers,
    )

    train_lr = float(lr if lr is not None else (RESNET_SPEC.lr_pretrained if pretrained else RESNET_SPEC.lr))
    ckpt_path = (
        resolve_osr_ckpt(ckpt, cfg.unknown_folder_id, pretrained=pretrained)
        if eval_only
        else str(checkpoint_dir() / osr_ckpt_name(cfg.unknown_folder_id, pretrained=pretrained))
    )

    split = open_train.split_osr_dataset(x_all, y_all, cfg)
    loaders = open_train.build_osr_loaders(split, cfg)

    if verbose:
        init_tag = "ImageNet fine-tune" if pretrained else "scratch"
        print(
            f"\n{'=' * 60}\n"
            f"[ResNet-18+MSP OSR] | unknown=folder {cfg.unknown_folder_id} ({split['unknown_name']}) "
            f"| known indices={cfg.known_class_indices}\n"
            f"  total={split['n_total']} | closed pool={split['n_closed_pool']} | unknown pool={split['n_unknown_pool']}\n"
            f"{init_tag} | batch={cfg.batch_size} | ASL single-head lr={train_lr:g} | "
            f"warmup={RESNET_SPEC.warmup_epochs} | early_stop={cfg.early_stop_patience}\n"
            f"{seed_status_line(cfg.seed, deterministic=deterministic)} | "
            f"rejection=MSP(1−max σ)\n"
            f"ckpt={ckpt_path}",
            flush=True,
        )
        if split["n_unknown_pool"] < 5:
            print(f"warn: unknown pool n={split['n_unknown_pool']}", flush=True)

    best_f1 = 0.0
    trained = not eval_only

    if eval_only:
        state = torch_load_checkpoint(ckpt_path, "cpu")
        model, _ = prepare_model_on_device(
            cfg,
            lambda: bl_model.build_baseline("resnet18", num_classes=cfg.num_classes, pretrained=pretrained),
            deterministic=deterministic,
            reset_seed=True,
        )
        model.load_state_dict(state["model_state_dict"])
        best_f1 = float(state.get("best_val_micro_f1", 0.0))
        if verbose:
            print(f"eval-only | best_val_μF1={best_f1:.4f}", flush=True)
    else:
        model, _ = prepare_model_on_device(
            cfg,
            lambda: bl_model.build_baseline("resnet18", num_classes=cfg.num_classes, pretrained=pretrained),
            deterministic=deterministic,
            reset_seed=True,
        )
        if verbose:
            print(
                f"params: {core.count_parameters(model)} | {bl_model.describe_architecture(model)}",
                flush=True,
            )
        best_f1 = bl_train.train_baseline_multilabel(
            model,
            loaders["train"],
            loaders["val"],
            cfg,
            lr=train_lr,
            warmup_epochs=RESNET_SPEC.warmup_epochs,
            verbose=verbose,
        )

    metrics = evaluate_resnet_msp_osr(
        model,
        loaders["test"],
        loaders["unknown"],
        loaders["val"],
        cfg,
    )

    if trained:
        save_osr_checkpoint(
            ckpt_path,
            model=model,
            cfg=cfg,
            best_f1=best_f1,
            metrics=metrics,
            pretrained=pretrained,
        )
        if verbose:
            print(f"saved: {ckpt_path}", flush=True)
    if verbose:
        print(
            f"[closed-set Test] μF1={metrics['closed_micro_f1']:.4f}",
            flush=True,
        )
        print(
            f"[MSP] closed={metrics['msp_closed_mean']:.4f} "
            f"unknown={metrics['msp_unknown_mean']:.4f} "
            f"gap={metrics['msp_gap_mean']:+.4f}",
            flush=True,
        )
        print(f"[open-set AUROC] MSP = {metrics['auroc_msp']:.4f}", flush=True)

    return {
        "unknown_folder_id": cfg.unknown_folder_id,
        "unknown_name": split["unknown_name"],
        "best_val_micro_f1": best_f1,
        **metrics,
        "ckpt": ckpt_path,
        "trained": int(trained),
    }


def format_table46_summary(rows: list[dict[str, Any]]) -> str:
    by_id = {int(r["unknown_folder_id"]): r for r in rows}
    lines = ["\nTable VI ResNet-18+MSP AUROC (vs our method primary metric)", "Exp | Unknown | Ours | ResNet+MSP"]
    for exp, uid in TABLE46_EXPERIMENTS.items():
        ours_name, ours_val = TABLE46_OURS_METRIC[exp]
        row = by_id.get(uid)
        msp = row["auroc_msp"] if row else float("nan")
        unk = row["unknown_name"] if row else open_train.GC10_UNKNOWN_NAMES.get(uid, f"folder{uid}")
        lines.append(
            f"  {exp} | {unk} | {ours_name} {ours_val:.4f} | MSP {msp:.4f}"
        )
    return "\n".join(lines)
