"""Shared pipeline for closed-set / open-set / baselines: splits, repro, reports, OSR unknown parsing."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import core, metrics, open_train
from .runtime import (
    apply_training_repro,
    print_repro_banner,
    seed_status_line,
    set_run_device,
    set_seed,
    split_indices,
)

_PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ClosedSplit:
    """GC10 closed-set 60/20/20 split + DataLoaders."""

    x_all: np.ndarray
    y_all: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def checkpoint_dir() -> Path:
    d = _PKG_ROOT / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = _PKG_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_closed_split(cfg: core.Config) -> ClosedSplit:
    x_all, y_all = core.prepare_full_dataset(cfg)
    tr_idx, va_idx, te_idx, _ = split_indices(
        y_all,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )
    return ClosedSplit(
        x_all=x_all,
        y_all=y_all,
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=te_idx,
        train_loader=core.make_loader(x_all[tr_idx], y_all[tr_idx], cfg, train=True),
        val_loader=core.make_loader(x_all[va_idx], y_all[va_idx], cfg, train=False),
        test_loader=core.make_loader(x_all[te_idx], y_all[te_idx], cfg, train=False),
    )


def load_gc10_tensors(cfg: core.Config) -> tuple[np.ndarray, np.ndarray]:
    """Load full GC10 once (10-dim labels) for open-set / baseline batch runs."""
    x_all, y_all = core.prepare_full_dataset(cfg)
    if y_all.shape[1] != 10:
        raise ValueError(f"GC10 OSR requires 10-dim labels, got {y_all.shape}")
    return x_all, y_all


def init_training_repro(
    cfg: core.Config,
    args: argparse.Namespace,
    *,
    script: str,
    default_deterministic: bool = False,
) -> bool:
    """apply_training_repro + banner + set_seed; shared by all entry points."""
    det = apply_training_repro(cfg, args, default_deterministic=default_deterministic)
    print_repro_banner(det, cfg, script=script)
    set_seed(cfg.seed, deterministic=det)
    return det


def prepare_model_on_device(
    cfg: core.Config,
    build_fn: Callable[[], nn.Module],
    *,
    deterministic: bool,
    reset_seed: bool = True,
) -> tuple[nn.Module, torch.device]:
    """Reset seed after data load → build model → move to device (reproducible init)."""
    if reset_seed:
        set_seed(cfg.seed, deterministic=deterministic)
    model = build_fn()
    model, dev = core.move_model_with_fallback(
        model,
        use_amp=cfg.use_amp,
        use_channels_last=cfg.use_channels_last,
    )
    set_run_device(dev)
    return model, dev


def print_multilabel_paper_report(
    report: dict[str, Any],
    *,
    best_train_val: float,
    cfg: core.Config,
    headline: str = "Closed-set main results",
) -> None:
    proto = metrics.eval_protocol_label(cfg)
    print("[Metric breakdown]", flush=True)
    print(
        metrics.format_multilabel_eval_report(report, best_train_val=best_train_val),
        flush=True,
    )
    print(
        f"{headline} → Val μF1={report['cal_val']['micro_f1']:.4f} | "
        f"Test μF1={report['cal_test']['micro_f1']:.4f}  ({proto})",
        flush=True,
    )


def write_csv_rows(rows: list[dict], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary written to: {out.resolve()}", flush=True)
    return out


def parse_unknown_folder_ids_str(text: str) -> tuple[int, ...]:
    """Parse multi-unknown folder IDs like ``4,6`` (IDs 1–10)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("unknown-classes is empty")
    ids = tuple(sorted({int(p.strip()) for p in raw.split(",") if p.strip()}))
    if not ids or any(u < 1 or u > 10 for u in ids):
        raise ValueError(f"invalid unknown-classes: {text!r} (comma-separated IDs 1–10)")
    return ids


def resolve_osr_unknown_ids(args: argparse.Namespace) -> list[int]:
    multi = (getattr(args, "unknown_classes", "") or "").strip()
    if multi:
        return list(parse_unknown_folder_ids_str(multi))
    if getattr(args, "unknown_all", False):
        return list(range(1, 11))
    if getattr(args, "experiment", ""):
        return [open_train.OSR_EXPERIMENTS[str(args.experiment).upper()]]
    if getattr(args, "unknown_class", None) is not None:
        return [int(args.unknown_class)]
    raise ValueError("specify --unknown-class, --unknown-classes, --unknown-all, or --experiment")


def resolve_osr_unknown_specs(args: argparse.Namespace) -> list[int | tuple[int, ...]]:
    """Single run spec: int for one unknown, tuple for multi (e.g. C+D → (4, 6))."""
    multi = (getattr(args, "unknown_classes", "") or "").strip()
    if multi:
        return [parse_unknown_folder_ids_str(multi)]
    return resolve_osr_unknown_ids(args)


def validate_osr_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Mutually exclusive open-set CLI checks (open.py / exp.py msp)."""
    if getattr(args, "closed_ckpt", "") and getattr(args, "init_closed", ""):
        parser.error("--closed-ckpt and --init-closed cannot be used together")
    if getattr(args, "no_proto", False) and getattr(args, "proto", False):
        parser.error("--proto and --no-proto cannot be used together")
    if getattr(args, "closed_ckpt", ""):
        args.eval_only = True
    unk_multi = (getattr(args, "unknown_classes", "") or "").strip()
    if unk_multi:
        if getattr(args, "unknown_all", False):
            parser.error("--unknown-classes and --unknown-all cannot be used together")
        if getattr(args, "unknown_class", None) is not None:
            parser.error("--unknown-classes and --unknown-class cannot be used together")
        if getattr(args, "experiment", ""):
            parser.error("--unknown-classes and --experiment cannot be used together")
    if not getattr(args, "unknown_all", False):
        if (
            getattr(args, "unknown_class", None) is None
            and not getattr(args, "experiment", "")
            and not unk_multi
        ):
            parser.error("specify --unknown-class, --unknown-classes, --unknown-all, or --experiment")
        return
    if getattr(args, "unknown_class", None) is not None:
        parser.error("--unknown-all and --unknown-class cannot be used together")
    if getattr(args, "experiment", ""):
        parser.error("--unknown-all and --experiment cannot be used together")
    if getattr(args, "ckpt", ""):
        parser.error("--unknown-all cannot be used with --ckpt")
    if getattr(args, "closed_ckpt", ""):
        parser.error("--unknown-all and --closed-ckpt cannot be used together")
