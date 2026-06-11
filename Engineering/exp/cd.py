"""C+D dual-unknown open-set (patch ID=4 + pitted surface ID=6, 8-class closed training)."""

from __future__ import annotations

import argparse

from exp.boot import ensure_root_on_path

ensure_root_on_path()

from utils import open_train
from utils.open_cli import run_osr_once
from utils.pipeline import init_training_repro, load_gc10_tensors, results_dir, write_csv_rows
from utils.runtime import add_repro_cli_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-unknown OSR: exp C pitted surface (6) + exp D patch (4), 8-class closed training",
    )
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("-b", "--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--proto", action="store_true")
    p.add_argument("--no-proto", action="store_true")
    p.add_argument("--no-decouple", action="store_true")
    p.add_argument("--no-dog", action="store_true")
    p.add_argument("--no-aux", action="store_true")
    add_repro_cli_args(p)
    args = p.parse_args(argv)
    if args.proto and args.no_proto:
        p.error("--proto and --no-proto are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    peek = open_train.OSRConfig()
    peek.apply_defaults()
    peek.data_root = str(args.data_root).strip()
    if args.seed is not None:
        peek.seed = int(args.seed)
    init_training_repro(peek, args, script="OSR C+D dual-unknown")

    cfg_load = open_train.OSRConfig()
    cfg_load.data_root = peek.data_root
    cfg_load.seed = peek.seed
    x_all, y_all = load_gc10_tensors(cfg_load)

    ids = open_train.OSR_CD_DUAL_UNKNOWN
    print(
        f"Dual-unknown | folder IDs {list(ids)} "
        f"(D patch={ids[0]}, C pitted={ids[1]}) | 8 known classes",
        flush=True,
    )
    row = run_osr_once(args, None, x_all, y_all, unknown_folder_ids=ids)
    suffix = "eval" if args.eval_only else "train"
    write_csv_rows([row], results_dir() / f"osr_dog_v2_unknown4_6_{suffix}.csv")
