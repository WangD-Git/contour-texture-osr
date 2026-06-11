"""Experiment A ablation (Table IV: none → DoG only → DoG+L_dec)."""

from __future__ import annotations

import argparse
from datetime import datetime

from exp.boot import ensure_root_on_path

ensure_root_on_path()

from utils import open_train
from utils.open_cli import run_osr_once
from utils.pipeline import init_training_repro, load_gc10_tensors, results_dir, write_csv_rows

EXPERIMENT_A_ID = int(open_train.OSR_EXPERIMENTS["A"])

ABLATIONS: tuple[tuple[str, str, dict], ...] = (
    (
        "no_ldec_dog",
        "1. none",
        {"ablation_tag": "no_ldec_dog", "no_decouple": True, "no_dog": True},
    ),
    (
        "no_ldec",
        "2. DoG only",
        {"ablation_tag": "no_ldec", "no_decouple": True},
    ),
    (
        "full",
        "3. DoG+L_dec",
        {"ablation_tag": "full"},
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table IV ablation: experiment A (inclusion)")
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-only", action="store_true", help="evaluate existing checkpoints only")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="run selected keys only, comma-separated: no_ldec_dog,no_ldec,full",
    )
    p.add_argument("--deterministic", action="store_true", help="strict reproducibility (default: seed only)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 44)")
    p.add_argument("--out-csv", type=str, default="")
    return p.parse_args(argv)


def _select_ablations(raw: str) -> tuple[tuple[str, str, dict], ...]:
    text = raw.strip().lower()
    if not text:
        return ABLATIONS
    keys = {k.strip() for k in text.split(",") if k.strip()}
    picked = tuple(row for row in ABLATIONS if row[0] in keys)
    unknown = keys - {row[0] for row in picked}
    if unknown:
        valid = ", ".join(row[0] for row in ABLATIONS)
        raise SystemExit(f"unknown --only key(s): {sorted(unknown)}; valid: {valid}")
    if not picked:
        raise SystemExit("specify at least one --only key")
    return picked


def _build_run_args(base: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    d = {
        "experiment": "",
        "unknown_class": EXPERIMENT_A_ID,
        "unknown_all": False,
        "unknown_classes": "",
        "data_root": base.data_root,
        "batch_size": base.batch_size,
        "epochs": base.epochs,
        "lr": base.lr,
        "proto": False,
        "no_proto": True,
        "no_push": True,
        "no_decouple": False,
        "no_dog": False,
        "ablation_tag": "",
        "eval_only": base.eval_only,
        "ckpt": "",
        "init_closed": "",
        "closed_ckpt": "",
        "deterministic": base.deterministic,
        "no_amp": False,
        "workers": None,
        "no_aux": False,
        "seed": base.seed,
    }
    d.update(overrides)
    return argparse.Namespace(**d)


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 72, flush=True)
    print("Table IV ablation | exp A (inclusion) | none → DoG only → DoG+L_dec", flush=True)
    print("=" * 72, flush=True)
    print(
        f"{'config':<16} {'|cos|↓':>10} {'main H↑':>10} {'JS↑':>10} {'Test μF1':>10}",
        flush=True,
    )
    print("-" * 72, flush=True)
    for r in rows:
        print(
            f"{r['label']:<16} "
            f"{r.get('feature_cosine_mean', float('nan')):>10.4f} "
            f"{r.get('auroc_entropy_main', float('nan')):>10.4f} "
            f"{r.get('auroc_disagreement', float('nan')):>10.4f} "
            f"{r.get('closed_micro_f1', float('nan')):>10.4f}",
            flush=True,
        )
    print("=" * 72, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = _select_ablations(args.only)

    cfg_load = open_train.OSRConfig()
    cfg_load.data_root = args.data_root.strip() if args.data_root else "."
    if args.seed is not None:
        cfg_load.seed = int(args.seed)
    init_training_repro(cfg_load, args, script="open-set ablation")

    x_all, y_all = load_gc10_tensors(cfg_load)

    print(
        f"Exp A ablation | unknown={EXPERIMENT_A_ID} "
        f"({open_train.GC10_UNKNOWN_NAMES.get(EXPERIMENT_A_ID, '')}) "
        f"| N={len(x_all)} | eval_only={args.eval_only} | runs: {', '.join(k for k, _, _ in runs)}",
        flush=True,
    )

    rows_out: list[dict] = []
    for key, label, overrides in runs:
        print(f"\n[{label}] ({key})", flush=True)
        run_args = _build_run_args(args, overrides)
        row = run_osr_once(run_args, EXPERIMENT_A_ID, x_all, y_all)
        rows_out.append(
            {
                "key": key,
                "label": label,
                "feature_cosine_mean": row.get("feature_cosine_mean"),
                "auroc_entropy_main": row.get("auroc_entropy_main"),
                "closed_micro_f1": row.get("closed_micro_f1"),
                "auroc_disagreement": row.get("auroc_disagreement"),
                "ckpt": row.get("ckpt", ""),
            }
        )

    _print_table(rows_out)

    out = args.out_csv.strip()
    if not out:
        out = str(results_dir() / f"ablation_expA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    write_csv_rows(
        rows_out,
        out,
        fieldnames=[
            "key",
            "label",
            "feature_cosine_mean",
            "auroc_entropy_main",
            "auroc_disagreement",
            "closed_micro_f1",
            "ckpt",
        ],
    )
