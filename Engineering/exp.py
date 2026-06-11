#!/usr/bin/env python3
"""Paper experiments: baseline, msp, ablation, cd."""

from __future__ import annotations

import argparse
import sys

from exp.boot import ensure_root_on_path

ensure_root_on_path()

from exp import (
    run_baseline,
    run_ablation,
    run_cd,
    run_msp,
)

SUBCOMMANDS = {
    "baseline": run_baseline,
    "msp": run_msp,
    "ablation": run_ablation,
    "cd": run_cd,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: exp.py {baseline,msp,ablation,cd} [options]")
        raise SystemExit(0)

    name = sys.argv[1]
    if name not in SUBCOMMANDS:
        print(f"unknown subcommand: {name!r}\navailable: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        raise SystemExit(1)

    SUBCOMMANDS[name](sys.argv[2:])


if __name__ == "__main__":
    main()
