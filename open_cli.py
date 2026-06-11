"""DoG five-stage dual-stream open-set training/eval CLI."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import core, open_train
from .bridge import (
    DEFAULT_CLOSED_CKPT,
    build_closed_model_for_osr_eval,
    load_closed_ckpt_into_osr,
    resolve_closed_ckpt,
)
from .model import build_model
from .runtime import (
    add_osr_unknown_cli_args,
    add_repro_cli_args,
    apply_training_repro,
    seed_status_line,
    set_seed,
    torch_load_checkpoint,
)
from .pipeline import (
    checkpoint_dir,
    init_training_repro,
    load_gc10_tensors,
    prepare_model_on_device,
    resolve_osr_unknown_specs,
    results_dir,
    validate_osr_cli_args,
    write_csv_rows,
)

OSR_METRIC_KEYS = (
    "closed_micro_f1",
    "auroc_disagreement",
    "auroc_entropy",
    "auroc_entropy_main",
    "auroc_entropy_contour",
    "auroc_entropy_texture",
    "auroc_max_prob",
    "auroc_combined",
    "auroc_proto_dist",
    "entropy_closed_mean",
    "entropy_unknown_mean",
    "entropy_gap_mean",
    "entropy_main_gap_mean",
    "entropy_contour_gap_mean",
    "entropy_texture_gap_mean",
    "n_closed_eval",
    "n_unknown_eval",
)


def _resolve_ckpt_path(path: str) -> str:
    """Resolve OSR checkpoint path (training output)."""
    p = Path(path.strip())
    if p.is_file():
        return str(p.resolve())
    for cand in (p, Path.cwd() / p, checkpoint_dir() / p.name):
        if cand.is_file():
            return str(cand.resolve())
    return str(p)


def _save_osr_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    cfg: open_train.OSRConfig,
    best_f1: float,
    known_idx: list[int],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    banks: tuple[open_train.BranchPrototypes, open_train.BranchPrototypes, open_train.BranchPrototypes] | None,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict = {
        "model_state_dict": model.state_dict(),
        "best_val_micro_f1": best_f1,
        "config": asdict(cfg),
        "arch": "dual_dog_five_stage_osr",
        "known_class_indices": known_idx,
        "unknown_folder_id": cfg.unknown_folder_id,
        "train_idx": tr.astype(np.int64),
        "val_idx": va.astype(np.int64),
        "test_idx": te.astype(np.int64),
    }
    if metrics is not None:
        payload["osr_metrics"] = metrics
    if banks is not None:
        payload["bank_contour"] = banks[0].state_dict()
        payload["bank_texture"] = banks[1].state_dict()
        payload["bank_fusion"] = banks[2].state_dict()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def _infer_ablation_tag_from_flags(args: argparse.Namespace) -> str:
    """Infer ckpt suffix from --no-decouple / --no-dog / --no-aux to avoid overwriting defaults."""
    parts: list[str] = []
    if getattr(args, "no_decouple", False):
        parts.append("no_ldec")
    if getattr(args, "no_dog", False):
        parts.append("no_dog")
    if getattr(args, "no_aux", False):
        parts.append("no_aux")
    return "_".join(parts)


def _effective_ablation_tag(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "ablation_tag", "") or "").strip()
    return explicit or _infer_ablation_tag_from_flags(args)


def _unknown_ckpt_tag(cfg: open_train.OSRConfig) -> str:
    return "_".join(str(u) for u in cfg.effective_unknown_folder_ids())


def _make_cfg_from_args(
    args: argparse.Namespace,
    *,
    unknown_folder_id: int | None = None,
    unknown_folder_ids: tuple[int, ...] | None = None,
) -> open_train.OSRConfig:
    cfg = open_train.OSRConfig()
    cfg.apply_defaults()
    if unknown_folder_ids:
        cfg.unknown_folder_ids = tuple(sorted({int(u) for u in unknown_folder_ids}))
        cfg.unknown_folder_id = int(cfg.unknown_folder_ids[0])
    else:
        cfg.unknown_folder_id = int(unknown_folder_id)
        cfg.unknown_folder_ids = None
    if getattr(args, "proto", False):
        cfg.use_prototype_loss = True
    if getattr(args, "no_proto", False):
        cfg.use_prototype_loss = False
    if getattr(args, "no_push", False):
        cfg.use_push_loss = False
    if getattr(args, "no_decouple", False):
        cfg.use_decouple_loss = False
    if getattr(args, "no_dog", False):
        cfg.use_dog_prep = False
        cfg.use_dog_bal_loss = False
    if getattr(args, "no_aux", False):
        cfg.use_aux_loss = False
    if getattr(args, "workers", None) is not None:
        cfg.num_workers = max(0, int(args.workers))
    if getattr(args, "data_root", None):
        cfg.data_root = str(args.data_root).strip()
    if getattr(args, "batch_size", None):
        cfg.batch_size = max(1, int(args.batch_size))
    if getattr(args, "epochs", None):
        cfg.epochs = max(1, int(args.epochs))
    if getattr(args, "lr", None):
        cfg.lr = max(1e-8, float(args.lr))
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    stem = core.osr_ckpt_stem(cfg)
    uid_tag = _unknown_ckpt_tag(cfg)
    tag = _effective_ablation_tag(args)
    if tag:
        cfg.save_path = str(checkpoint_dir() / f"osr_{stem}_{tag}_unknown{uid_tag}_best.pth")
    else:
        cfg.save_path = str(checkpoint_dir() / f"osr_{stem}_unknown{uid_tag}_best.pth")
    init_closed = getattr(args, "init_closed", "") or ""
    if init_closed:
        cfg.closed_init_ckpt = init_closed.strip()
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GC10 open-set: DoG five-stage dual stream + entropy/JS rejection")
    add_osr_unknown_cli_args(p)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--proto", action="store_true", help="enable prototype pull/push loss (default off)")
    p.add_argument("--no-proto", action="store_true", help="explicitly disable prototype loss")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-decouple", action="store_true", help="disable L_dec decouple loss (default on)")
    p.add_argument("--no-dog", action="store_true", help="disable DoG frontend, grayscale 2-ch passthrough (w/o DoG)")
    p.add_argument("--no-aux", action="store_true", help="disable aux head L_aux")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--init-closed", type=str, default="")
    p.add_argument("--closed-ckpt", type=str, default="")
    add_repro_cli_args(p)
    args = p.parse_args()
    validate_osr_cli_args(p, args)
    return args


def _load_banks_from_ckpt(
    state: dict, cfg: open_train.OSRConfig
) -> tuple[open_train.BranchPrototypes, open_train.BranchPrototypes, open_train.BranchPrototypes] | None:
    keys = ("bank_contour", "bank_texture", "bank_fusion")
    if not all(k in state for k in keys):
        return None
    banks = open_train.build_prototype_banks(cfg, proto_ema=cfg.proto_ema)
    for bank, key in zip(banks, keys):
        bank.load_state_dict(state[key])
    return banks


def _print_entropy_report(metrics: dict[str, float]) -> None:
    print("\n[Entropy stats]", flush=True)
    for label, key in (
        ("main-head entropy", "entropy_main"),
        ("contour aux entropy", "entropy_contour"),
        ("texture aux entropy", "entropy_texture"),
        ("fused max(main,contour,texture)", "entropy"),
    ):
        cm = metrics.get(f"{key}_closed_mean", float("nan"))
        um = metrics.get(f"{key}_unknown_mean", float("nan"))
        gap = metrics.get(f"{key}_gap_mean", float("nan"))
        au = metrics.get(f"auroc_{key}", float("nan"))
        print(
            f"  {label:16s} | closed={cm:.4f} unknown={um:.4f} gap={gap:+.4f} | AUROC={au:.4f}",
            flush=True,
        )


def run_osr_once(
    args: argparse.Namespace,
    unknown_folder_id: int | None,
    x_all: np.ndarray,
    y_all: np.ndarray,
    *,
    unknown_folder_ids: tuple[int, ...] | None = None,
) -> dict[str, float]:
    if unknown_folder_ids:
        cfg = _make_cfg_from_args(args, unknown_folder_ids=unknown_folder_ids)
    else:
        cfg = _make_cfg_from_args(args, unknown_folder_id=int(unknown_folder_id))
    cfg.apply_osr_model_dims()
    det = apply_training_repro(cfg, args, default_deterministic=False)
    set_seed(cfg.seed, deterministic=det)
    loss_flags = (
        f"L_aux={'on' if cfg.use_aux_loss else 'off'} | "
        f"L_dec={'on' if cfg.use_decouple_loss else 'off'} | "
        f"DoG={'on' if cfg.use_dog_prep else 'off'} | "
        f"proto={'on' if cfg.use_prototype_loss else 'off'}"
    )

    split = open_train.split_osr_dataset(x_all, y_all, cfg)
    known_idx = cfg.known_class_indices
    loaders = open_train.build_osr_loaders(split, cfg)
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    unk_ids = cfg.effective_unknown_folder_ids()
    print(
        f"\n{'=' * 60}\n"
        f"OSR[DoG v2] | unknown=folders {list(unk_ids)} ({split['unknown_name']}) "
        f"| {len(known_idx)} known classes indices={known_idx}\n"
        f"  {loss_flags}\n"
        f"  {seed_status_line(cfg.seed, deterministic=det)} | workers={cfg.num_workers} | "
        f"amp={cfg.use_amp} | early_stop={cfg.early_stop_patience}",
        flush=True,
    )
    print(
        f"  total={split['n_total']} | closed pool={split['n_closed_pool']} | unknown pool={split['n_unknown_pool']}",
        flush=True,
    )
    if split["n_unknown_pool"] < 5:
        print(f"warn: unknown pool n={split['n_unknown_pool']}", flush=True)

    use_closed_only = bool(getattr(args, "closed_ckpt", "") or "")
    closed_ckpt_path = (getattr(args, "closed_ckpt", "") or "").strip()
    if use_closed_only and not closed_ckpt_path:
        closed_ckpt_path = DEFAULT_CLOSED_CKPT

    preload_state: dict | None = None
    set_seed(cfg.seed, deterministic=det)

    if use_closed_only:
        ckpt_path = resolve_closed_ckpt(closed_ckpt_path)
        preload_state = torch_load_checkpoint(ckpt_path, "cpu")
        pool = core.apply_pool_backend_from_ckpt(cfg, preload_state)
        print(f"closed-set direct eval | ckpt={ckpt_path} | pool={core.pool_backend_label(pool)}", flush=True)
        model, dev = prepare_model_on_device(
            cfg,
            lambda: build_closed_model_for_osr_eval(cfg, known_idx, use_blur_pool=pool),
            deterministic=det,
            reset_seed=False,
        )
    elif args.eval_only:
        ckpt_path = _resolve_ckpt_path(args.ckpt or cfg.save_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        preload_state = torch_load_checkpoint(ckpt_path, "cpu")
        pool = core.apply_pool_backend_from_ckpt(cfg, preload_state)
        print(f"eval-only | ckpt={ckpt_path} | pool={core.pool_backend_label(pool)}", flush=True)
        model, dev = prepare_model_on_device(
            cfg,
            lambda: build_model(cfg),
            deterministic=det,
            reset_seed=True,
        )
    else:
        init_path = (getattr(args, "init_closed", "") or cfg.closed_init_ckpt or "").strip()
        if init_path:
            try:
                init_resolved = resolve_closed_ckpt(init_path)
                preload_state = torch_load_checkpoint(init_resolved, "cpu")
                pool = core.apply_pool_backend_from_ckpt(cfg, preload_state)
                print(
                    f"pool matched from init-closed={core.pool_backend_label(pool)}: {init_resolved}",
                    flush=True,
                )
            except FileNotFoundError:
                print(f"warn: --init-closed missing: {init_path}", flush=True)
        model, dev = prepare_model_on_device(
            cfg,
            lambda: build_model(cfg),
            deterministic=det,
            reset_seed=True,
        )

    banks = None
    best_f1 = 0.0
    trained = not args.eval_only and not use_closed_only

    if use_closed_only:
        assert preload_state is not None
        model.model.load_state_dict(preload_state["model_state_dict"])
        print(f"loaded closed-set DoG v2 (10-class→9-class): {ckpt_path}", flush=True)
    elif args.eval_only:
        assert preload_state is not None
        model.load_state_dict(preload_state["model_state_dict"])
        best_f1 = float(preload_state.get("best_val_micro_f1", 0))
        banks = _load_banks_from_ckpt(preload_state, cfg)
    else:
        print(f"params: {core.count_parameters(model)} | train → {cfg.save_path}", flush=True)
        init_path = (cfg.closed_init_ckpt or "").strip()
        if init_path:
            try:
                init_resolved = resolve_closed_ckpt(init_path)
                miss = load_closed_ckpt_into_osr(model, init_resolved, known_idx, device=dev)
                print(f"closed-set warm start: {init_resolved} | missing keys={len(miss)}", flush=True)
            except FileNotFoundError:
                print(f"warn: --init-closed missing: {init_path}", flush=True)
        best_f1, banks = open_train.train_osr_classifier(model, loaders["train"], loaders["val"], cfg)

    if banks is None and cfg.use_prototype_loss and not use_closed_only:
        banks = open_train.build_prototype_banks(cfg, proto_ema=cfg.proto_ema)

    metrics = open_train.evaluate_osr(
        model,
        loaders["test"],
        loaders["unknown"],
        cfg,
        banks,
        calib_loader=loaders["val"],
    )
    print(f"[closed-set Test] μF1={metrics['closed_micro_f1']:.4f}", flush=True)
    print(f"[features] contour–texture L1 mean |cos| = {metrics.get('feature_cosine_mean', float('nan')):.4f}", flush=True)
    _print_entropy_report(metrics)
    print("\n[open-set AUROC]", flush=True)
    for k in (
        "disagreement",
        "entropy_main",
        "entropy_contour",
        "entropy_texture",
        "entropy",
        "max_prob",
        "combined",
    ):
        print(f"  AUROC_{k:12s} = {metrics[f'auroc_{k}']:.4f}", flush=True)
    proto_au = metrics.get("auroc_proto_dist", float("nan"))
    if np.isnan(proto_au):
        print("  AUROC_proto_dist   = (skipped)", flush=True)
    else:
        print(f"  AUROC_proto_dist   = {proto_au:.4f}", flush=True)

    if trained:
        _save_osr_checkpoint(
            cfg.save_path,
            model=model,
            cfg=cfg,
            best_f1=best_f1,
            known_idx=known_idx,
            tr=tr,
            va=va,
            te=te,
            banks=banks,
            metrics=metrics,
        )
        print(f"saved: {cfg.save_path}", flush=True)

    return {
        "unknown_folder_id": cfg.unknown_folder_id,
        "unknown_folder_ids": ",".join(str(u) for u in unk_ids),
        "unknown_name": split["unknown_name"],
        "n_known_classes": len(known_idx),
        "use_prototype_loss": cfg.use_prototype_loss,
        "feature_cosine_mean": metrics.get("feature_cosine_mean", float("nan")),
        "best_val_micro_f1": best_f1,
        **{k: metrics.get(k, float("nan")) for k in OSR_METRIC_KEYS},
        "ckpt": cfg.save_path,
    }


def main() -> None:
    args = parse_args()
    peek = open_train.OSRConfig()
    peek.apply_defaults()
    if args.data_root:
        peek.data_root = args.data_root.strip()
    if args.seed is not None:
        peek.seed = int(args.seed)
    init_training_repro(peek, args, script="open-set OSR")

    cfg_load = open_train.OSRConfig()
    cfg_load.data_root = peek.data_root
    cfg_load.seed = peek.seed
    x_all, y_all = load_gc10_tensors(cfg_load)

    rows: list[dict] = []
    for spec in resolve_osr_unknown_specs(args):
        eval_args = argparse.Namespace(**vars(args))
        eval_args.ckpt = "" if args.unknown_all else args.ckpt
        if isinstance(spec, tuple):
            rows.append(run_osr_once(eval_args, None, x_all, y_all, unknown_folder_ids=spec))
        else:
            rows.append(run_osr_once(eval_args, int(spec), x_all, y_all))

    if len(rows) > 1:
        suffix = "eval" if args.eval_only else "train"
        ab_tag = _effective_ablation_tag(args)
        csv_stem = f"osr_dog_v2_{ab_tag}_all_{suffix}" if ab_tag else f"osr_dog_v2_all_{suffix}"
        write_csv_rows(rows, results_dir() / f"{csv_stem}.csv")
    elif len(rows) == 1 and (getattr(args, "unknown_classes", "") or "").strip():
        suffix = "eval" if args.eval_only else "train"
        tag = rows[0].get("unknown_folder_ids", "multi").replace(",", "_")
        write_csv_rows(rows, results_dir() / f"osr_dog_v2_unknown{tag}_{suffix}.csv")
