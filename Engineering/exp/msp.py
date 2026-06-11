"""Open-set MSP baseline (Table VI: ResNet-18 scratch + MSP)."""

from __future__ import annotations

import argparse
from datetime import datetime

from exp.boot import ensure_root_on_path

ensure_root_on_path()

from utils import bl_open, open_train
from utils.pipeline import (
    init_training_repro,
    load_gc10_tensors,
    resolve_osr_unknown_ids,
    results_dir,
    validate_osr_cli_args,
    write_csv_rows,
)
from utils.runtime import add_osr_unknown_cli_args, add_repro_cli_args

CSV_FIELDS = (
    "unknown_folder_id",
    "unknown_name",
    "best_val_micro_f1",
    "closed_micro_f1",
    "auroc_msp",
    "msp_closed_mean",
    "msp_unknown_mean",
    "msp_gap_mean",
    "n_closed_eval",
    "n_unknown_eval",
    "ckpt",
    "trained",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GC10 open-set ResNet-18+MSP baseline (Table VI)")
    add_osr_unknown_cli_args(p)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint path for eval-only")
    p.add_argument("--pretrained", action="store_true", help="ImageNet pretrained (not main-table setting)")
    p.add_argument("--out-csv", type=str, default="")
    add_repro_cli_args(p)
    args = p.parse_args(argv)
    validate_osr_cli_args(p, args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    peek = open_train.OSRConfig()
    peek.data_root = args.data_root.strip()
    if args.seed is not None:
        peek.seed = int(args.seed)
    det = init_training_repro(peek, args, script="open-set ResNet+MSP baseline")

    x_all, y_all = load_gc10_tensors(peek)

    rows: list[dict] = []
    for uid in resolve_osr_unknown_ids(args):
        ckpt = "" if args.unknown_all else args.ckpt
        rows.append(
            bl_open.run_resnet18_osr_once(
                unknown_folder_id=uid,
                x_all=x_all,
                y_all=y_all,
                data_root=args.data_root,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                eval_only=args.eval_only,
                ckpt=ckpt,
                pretrained=args.pretrained,
                seed=peek.seed,
                deterministic=det,
                num_workers=peek.num_workers,
            )
        )

    if len(rows) > 1:
        suffix = "eval" if args.eval_only else "train"
        tag = "_pretrained" if args.pretrained else ""
        out = args.out_csv.strip()
        if not out:
            out = str(
                results_dir()
                / f"baseline_osr_resnet18_msp_all_{suffix}{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        write_csv_rows(rows, out, fieldnames=list(CSV_FIELDS))

    if args.experiment or args.unknown_all:
        print(bl_open.format_table46_summary(rows), flush=True)
