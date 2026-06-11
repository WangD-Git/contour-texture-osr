"""GC10/NEU closed-set classification training CLI."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

from . import closed_train, core, metrics
from .model import build_model
from .neu import apply_neu_preset
from .pipeline import (
    build_closed_split,
    init_training_repro,
    prepare_model_on_device,
    print_multilabel_paper_report,
)
from .runtime import add_repro_cli_args, seed_status_line, torch_load_checkpoint


def parse_args() -> tuple[argparse.Namespace, core.Config]:
    p = argparse.ArgumentParser(description="DoG contour-texture five-stage dual-stream (GC10/NEU)")
    p.add_argument("--dataset", choices=("gc10", "neu"), default="gc10")
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument(
        "--max-pool",
        action="store_true",
        help="use MaxPool for downsampling (default: anti-aliasing BlurPool)",
    )
    p.add_argument("--no-tta", action="store_true", help="disable TTA in final eval (default: T/τ+TTA)")
    add_repro_cli_args(p)
    p.add_argument(
        "--early-stop",
        type=int,
        default=None,
        metavar="N",
        help="stop after N epochs with no val μF1 gain (default 30; 0=run full --epochs)",
    )
    p.add_argument(
        "--adaptive-loss",
        action="store_true",
        help="enable per-batch dynamic aux/decouple weights (default off)",
    )
    args = p.parse_args()

    cfg = core.Config()
    core.apply_preset(cfg)
    if args.max_pool:
        cfg.use_blur_pool = False
    if args.dataset == "neu":
        apply_neu_preset(cfg)
    if args.data_root:
        cfg.data_root = args.data_root.strip()
    if args.batch_size:
        cfg.batch_size = max(1, int(args.batch_size))
    if args.epochs:
        cfg.epochs = max(1, int(args.epochs))
    if args.lr:
        cfg.lr = max(1e-8, float(args.lr))
    if args.no_tta:
        cfg.use_tta_eval = False
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.workers is not None:
        cfg.num_workers = max(0, int(args.workers))
    if args.early_stop is not None:
        cfg.early_stop_patience = max(0, int(args.early_stop))
    if args.adaptive_loss:
        cfg.use_adaptive_loss_balance = True
    return args, cfg


def main() -> None:
    args, cfg = parse_args()
    is_single = args.dataset == "neu"
    deterministic = init_training_repro(cfg, args, script="closed-set")

    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
    split = build_closed_split(cfg)

    pool_tag = core.pool_backend_label(cfg.use_blur_pool)
    print(
        f"DualStream | {args.dataset} | DoG+high-pass → five-stage v2 (8-16-32) "
        f"contour 5×5 / texture 3×3 | main=8+32 cross-branch | aux=8+32 per branch | DSConv+{pool_tag} | N={len(split.x_all)}",
        flush=True,
    )
    print(seed_status_line(cfg.seed, deterministic=deterministic), flush=True)
    if not is_single:
        print(
            f"model selection: val μF1 @τ=0.5 no TTA | early_stop={cfg.early_stop_patience} | "
            f"channels_last={cfg.use_channels_last} | adaptive_loss={cfg.use_adaptive_loss_balance} | "
            f"post-train eval: plain + paper ({metrics.eval_protocol_label(cfg)})",
            flush=True,
        )

    ckpt_state = None
    if args.eval_only:
        ckpt_path = args.ckpt or cfg.save_path
        ckpt_state = torch_load_checkpoint(ckpt_path, "cpu")
        inferred = core.apply_pool_backend_from_ckpt(cfg, ckpt_state)
        pool_tag = core.pool_backend_label(inferred)
        print(f"eval-only | pool matched from ckpt={pool_tag} | {ckpt_path}", flush=True)

    model, _ = prepare_model_on_device(cfg, lambda: build_model(cfg), deterministic=deterministic)
    print(f"params: {core.count_parameters(model)} | ckpt={cfg.save_path}", flush=True)

    best_metric = 0.0
    if args.eval_only:
        assert ckpt_state is not None
        model.load_state_dict(ckpt_state["model_state_dict"])
        best_metric = float(ckpt_state.get("best_val_micro_f1", ckpt_state.get("best_val_accuracy", 0)))
    elif is_single:
        best_metric = closed_train.train_singlelabel_classifier(
            model, split.train_loader, split.val_loader, cfg
        )
    else:
        best_metric = closed_train.train_multilabel_classifier(
            model, split.train_loader, split.val_loader, cfg
        )

    if is_single:
        val_m, test_m = closed_train.evaluate_singlelabel(model, split.val_loader, split.test_loader, cfg)
        print(f"[Val] acc={val_m['accuracy']:.4f}", flush=True)
        print(f"[Test] acc={test_m['accuracy']:.4f}", flush=True)
    if not args.eval_only:
        payload = {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "arch": "dual_dog_five_stage",
            "widths": list(cfg.channel_widths or ()),
        }
        if is_single:
            payload["best_val_accuracy"] = best_metric
        else:
            payload["best_val_micro_f1"] = best_metric
        torch.save(payload, cfg.save_path)
        print(f"saved: {cfg.save_path}", flush=True)

    if not is_single:
        report = closed_train.evaluate_multilabel(model, split.val_loader, split.test_loader, cfg)
        print_multilabel_paper_report(report, best_train_val=best_metric, cfg=cfg)
