"""Closed-set CNN baselines (Table II: ResNet-18 / MobileNetV3 scratch)."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from datetime import datetime

import torch

from exp.boot import ensure_root_on_path

ensure_root_on_path()

from utils import bl_model, bl_train, core, metrics
from utils.pipeline import (
    build_closed_split,
    checkpoint_dir,
    init_training_repro,
    prepare_model_on_device,
    print_multilabel_paper_report,
    results_dir,
    write_csv_rows,
)
from utils.runtime import add_repro_cli_args, seed_status_line, torch_load_checkpoint

DEFAULT_MODELS = ["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]
ALL_MODEL_KEYS = sorted(bl_model.BASELINE_BY_KEY.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GC10 closed-set CNN baselines (ResNet-18 / MobileNetV3)")
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"comma-separated; all=Table II three models; options: {', '.join(ALL_MODEL_KEYS)}",
    )
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--pretrained", action="store_true", help="ImageNet pretrained (not main-table setting)")
    p.add_argument("--no-tta", action="store_true", help="disable TTA in test eval")
    add_repro_cli_args(p)
    p.add_argument("--verify-arch", action="store_true", help="verify three model structures/param counts and exit")
    p.add_argument("--out-csv", type=str, default="")
    return p.parse_args(argv)


def _verify_and_exit() -> None:
    warns = bl_model.verify_table45_models()
    for k in bl_model.PAPER_PARAM_COUNTS:
        m = bl_model.build_baseline(k)
        print(f"[{k}] {bl_model.describe_architecture(m)}")
        print(f"  params={core.count_parameters(m)} paper={bl_model.PAPER_PARAM_COUNTS[k]}")
    if warns:
        for w in warns:
            print("WARN:", w)
        raise SystemExit(1)
    print("Table II: all three models pass structure/param check.")


def _resolve_models(raw: str) -> list[str]:
    text = raw.strip().lower()
    if text in ("", "all"):
        return list(DEFAULT_MODELS)
    keys = [k.strip().lower() for k in text.split(",") if k.strip()]
    unknown = [k for k in keys if k not in bl_model.BASELINE_BY_KEY]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; options: {ALL_MODEL_KEYS}")
    return keys


def _run_one(spec: bl_model.BaselineSpec, cfg: core.Config, args: argparse.Namespace) -> dict:
    os.makedirs("checkpoints", exist_ok=True)

    run_cfg = replace(cfg)
    run_cfg.use_channels_last = False
    if args.batch_size is None and spec.batch_size is not None:
        run_cfg.batch_size = spec.batch_size
    run_cfg.early_stop_patience = spec.early_stop_patience
    if args.no_tta:
        run_cfg.use_tta_eval = False

    det = init_training_repro(run_cfg, args, script=f"baseline {spec.key}")
    split = build_closed_split(run_cfg)

    if args.lr is not None:
        train_lr = float(args.lr)
    elif args.pretrained:
        if not spec.supports_pretrained:
            raise SystemExit(f"{spec.key} does not support --pretrained; use resnet18")
        train_lr = spec.lr_pretrained
    elif spec.lr is not None:
        train_lr = spec.lr
    else:
        train_lr = run_cfg.lr

    ckpt_path = str(checkpoint_dir() / bl_model.resolve_ckpt_name(spec, pretrained=args.pretrained))
    init_tag = "ImageNet fine-tune" if args.pretrained else "scratch"
    print(
        f"\n{'=' * 60}\n"
        f"[{spec.title}] | N={len(split.x_all)} | {init_tag} | 1ch gray stem\n"
        f"batch={run_cfg.batch_size} | ASL+AdamW lr={train_lr:g} | warmup={spec.warmup_epochs} | "
        f"early_stop={run_cfg.early_stop_patience} | seed={run_cfg.seed}\n"
        f"{seed_status_line(run_cfg.seed, deterministic=det)}\n"
        f"ckpt={ckpt_path}",
        flush=True,
    )

    model, _ = prepare_model_on_device(
        run_cfg,
        lambda: bl_model.build_baseline(spec.key, num_classes=run_cfg.num_classes, pretrained=args.pretrained),
        deterministic=det,
    )
    n_params = core.count_parameters(model)
    print(f"params: {n_params} (paper={spec.paper_params}) | {bl_model.describe_architecture(model)}", flush=True)

    best_val = 0.0
    if args.eval_only:
        state = torch_load_checkpoint(ckpt_path, "cpu")
        model.load_state_dict(state["model_state_dict"])
        best_val = float(state.get("best_val_micro_f1", 0.0))
        print(f"eval-only | best_val_μF1={best_val:.4f}", flush=True)
    else:
        best_val = bl_train.train_baseline_multilabel(
            model,
            split.train_loader,
            split.val_loader,
            run_cfg,
            lr=train_lr,
            warmup_epochs=spec.warmup_epochs,
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "config": asdict(run_cfg),
            "arch": spec.key,
            "baseline": spec.title,
            "pretrained": bool(args.pretrained),
            "train_lr": train_lr,
            "best_val_micro_f1": best_val,
        }
        torch.save(payload, ckpt_path)
        print(f"saved: {ckpt_path}", flush=True)

    report = metrics.evaluate_multilabel_report(model, split.val_loader, split.test_loader, run_cfg)
    print_multilabel_paper_report(
        report,
        best_train_val=best_val,
        cfg=run_cfg,
        headline="Table II closed-set baseline",
    )

    return {
        "model": spec.key,
        "title": spec.title,
        "params": n_params,
        "pretrained": int(args.pretrained),
        "train_lr": train_lr,
        "best_val_train_loop": best_val,
        "plain_val_micro_f1": report["plain_val"]["micro_f1"],
        "plain_test_micro_f1": report["plain_test"]["micro_f1"],
        "val_micro_f1": report["cal_val"]["micro_f1"],
        "test_micro_f1": report["cal_test"]["micro_f1"],
        "ckpt": ckpt_path,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify_arch:
        _verify_and_exit()
        return

    cfg = core.Config()
    core.apply_preset(cfg)
    cfg.data_root = args.data_root.strip() or "."
    if args.batch_size:
        cfg.batch_size = max(1, int(args.batch_size))
    if args.epochs:
        cfg.epochs = max(1, int(args.epochs))
    if args.lr:
        cfg.lr = max(1e-8, float(args.lr))
    if args.seed is not None:
        cfg.seed = int(args.seed)

    rows = [_run_one(bl_model.BASELINE_BY_KEY[key], cfg, args) for key in _resolve_models(args.models)]

    out_csv = args.out_csv.strip() or str(
        results_dir() / f"baselines_gc10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    write_csv_rows(rows, out_csv)
