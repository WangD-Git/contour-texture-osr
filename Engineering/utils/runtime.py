"""Runtime utilities: device, data paths, splits (classification training)."""

from __future__ import annotations

import argparse
import os
import random

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_PKG_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DEVICE = DEVICE

STRATIFIED_SPLIT_TRIES = 256
RANDOM_SUBSEED_MAX = 2**31 - 1
MIN_SAMPLES_FOR_THREE_WAY_SPLIT = 3


def set_run_device(device: torch.device) -> None:
    global RUN_DEVICE
    RUN_DEVICE = device


def add_repro_cli_args(parser: argparse.ArgumentParser) -> None:
    """seed / deterministic / workers / AMP — shared by closed-set, open-set, baselines."""
    parser.add_argument("--seed", type=int, default=None, help="random seed (default 44)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="strict reproducibility (cudnn deterministic + workers=0; default: seed only)",
    )
    parser.add_argument("--no-amp", action="store_true", help="disable mixed-precision AMP")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader num_workers (default 2)")


def add_osr_unknown_cli_args(parser: argparse.ArgumentParser) -> None:
    """Leave-one-unknown OSR: experiment / unknown-class / unknown-all / unknown-classes."""
    parser.add_argument("--experiment", type=str, default="", choices=("", "A", "B", "C", "D"))
    parser.add_argument("--unknown-class", type=int, default=None)
    parser.add_argument(
        "--unknown-classes",
        type=str,
        default="",
        help="multiple unknown folder IDs, comma-separated, e.g. 4,6 (Patches + Pitted surface)",
    )
    parser.add_argument("--unknown-all", action="store_true", help="run all 10 leave-one-unknown settings and write CSV")


def set_seed(seed: int = 44, *, deterministic: bool = False) -> None:
    """Fix Python/NumPy/torch RNG; tighten CUDA when deterministic=True."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


def seed_status_line(seed: int, *, deterministic: bool) -> str:
    cuda = torch.cuda.is_available()
    return (
        f"seed={seed} | deterministic={deterministic} | "
        f"cuda={cuda} | cudnn.deterministic={torch.backends.cudnn.deterministic} | "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark} | "
        f"workers={'0' if deterministic else '2'}"
    )


def apply_training_repro(cfg, args, *, default_deterministic: bool = False) -> bool:
    """Default: cfg.seed + set_seed. --deterministic sets workers=0, sync_h2d, cudnn deterministic."""
    workers_arg = getattr(args, "workers", None)
    det_flag = getattr(args, "deterministic", None)
    if det_flag is True:
        deterministic = True
    elif det_flag is False:
        deterministic = False
    else:
        deterministic = default_deterministic

    if getattr(args, "no_amp", False):
        cfg.use_amp = False

    if deterministic and workers_arg is None:
        cfg.num_workers = 0
    if deterministic:
        cfg.sync_h2d = True
    return deterministic


def print_repro_banner(deterministic: bool, cfg, *, script: str = "training") -> None:
    amp = "on" if cfg.use_amp else "off"
    if deterministic:
        print(
            f"{script} strict repro (--deterministic: cudnn deterministic + workers={cfg.num_workers} + AMP={amp})",
            flush=True,
        )
    else:
        print(
            f"{script} seed={cfg.seed}, workers={cfg.num_workers}, AMP={amp}",
            flush=True,
        )


def move_model_with_fallback(
    model: nn.Module,
    *,
    use_amp: bool = True,
    use_channels_last: bool = True,
) -> tuple[nn.Module, torch.device]:
    try:
        model = model.to(RUN_DEVICE)
        return model, RUN_DEVICE
    except RuntimeError as e:
        if RUN_DEVICE.type != "cuda":
            raise
        print(f"warn: CUDA fallback to CPU: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu = torch.device("cpu")
        return model.to(cpu), cpu


def torch_load_checkpoint(path: str, map_location: torch.device | str) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_cls_checkpoint_name(name: str) -> bool:
    n = name.lower()
    if n.startswith("gc10_"):
        return False
    return any(k in n for k in ("ification", "balanced", "classification", "cls"))


def _cls_ckpt_search_candidates(rel: Path) -> list[Path]:
    """Common checkpoint search paths (incl. wang/checkpoints)."""
    name = rel.name
    bases = (Path.cwd(), _PKG_ROOT, _PKG_ROOT.parent)
    out: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        for sub in ("", "checkpoints", "wang/checkpoints"):
            c = (base / sub / name) if sub else (base / rel)
            key = str(c)
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def resolve_cls_ckpt(ckpt_path: str | None = None) -> str:
    """Locate classification weights produced by train.py."""
    explicit = str(ckpt_path).strip() if ckpt_path else ""
    tried: list[str] = []
    if explicit:
        p = Path(explicit).expanduser()
        for c in (p, *_cls_ckpt_search_candidates(p)):
            tried.append(str(c))
            if not c.is_file():
                continue
            if not _is_cls_checkpoint_name(c.name):
                raise ValueError(f"--ckpt is not a known classification checkpoint name: {c.name}")
            return str(c.resolve())
        for root in (Path.cwd(), _PKG_ROOT, _PKG_ROOT.parent):
            if not root.exists():
                continue
            for hit in root.rglob(p.name):
                if hit.is_file() and _is_cls_checkpoint_name(hit.name):
                    return str(hit.resolve())
        raise FileNotFoundError(f"classification checkpoint not found: {explicit}")

    for root in (Path.cwd(), _PKG_ROOT, _PKG_ROOT.parent):
        if not root.exists():
            continue
        for hit in root.rglob("ification_balanced_best.pth"):
            if hit.is_file() and _is_cls_checkpoint_name(hit.name):
                return str(hit.resolve())
    for c in _cls_ckpt_search_candidates(Path("ification_balanced_best.pth")):
        tried.append(str(c))
        if c.is_file():
            return str(c.resolve())
    raise FileNotFoundError("classification checkpoint ification_balanced_best.pth not found")


def _gc10_label_dir(base: Path, dataset_name: str, label_dir_name: str) -> Path | None:
    """Return XML label dir if *base* is a valid GC10 data_root (parent of dataset folder)."""
    direct = base / dataset_name / label_dir_name
    if direct.is_dir():
        return direct
    return None


def _discover_gc10_data_root(
    base: Path,
    dataset_name: str,
    label_dir_name: str,
) -> str | None:
    """Resolve directory whose child is ``GC10-DET/lable/`` (XML multilabel protocol)."""
    hit = _gc10_label_dir(base, dataset_name, label_dir_name)
    if hit is not None:
        return str(base.resolve())

    # --data-root points directly at GC10-DET
    if base.name == dataset_name:
        alt = base / label_dir_name
        if alt.is_dir():
            return str(base.parent.resolve())

    if not base.is_dir():
        return None

    # Common layout: Engineering/ under repo root
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        hit = _gc10_label_dir(child, dataset_name, label_dir_name)
        if hit is not None:
            return str(child.resolve())
        if child.name == dataset_name and (child / label_dir_name).is_dir():
            return str(base.resolve())
    return None


def _extra_gc10_search_roots() -> list[Path]:
    names = ("Engineering",)
    anchors = (Path.cwd(), _PKG_ROOT, _PKG_ROOT.parent)
    out: list[Path] = []
    seen: set[str] = set()
    for anchor in anchors:
        if not anchor.is_dir():
            continue
        for name in names:
            p = (anchor / name).resolve()
            key = str(p)
            if key in seen or not p.is_dir():
                continue
            seen.add(key)
            out.append(p)
    return out


def resolve_data_root(
    data_root: str | None = None,
    *,
    dataset_name: str = "GC10-DET",
    label_dir_name: str = "lable",
) -> str:
    bases: list[Path] = []
    if data_root and str(data_root).strip():
        bases.append(Path(data_root).expanduser())
    bases.extend([Path.cwd(), _PKG_ROOT, _PKG_ROOT.parent])
    bases.extend(_extra_gc10_search_roots())
    tried: list[str] = []
    seen: set[str] = set()
    for base in bases:
        try:
            resolved = base.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)

        gc10_root = _discover_gc10_data_root(resolved, dataset_name, label_dir_name)
        if gc10_root is not None:
            return gc10_root

        neu_labels = resolved / dataset_name / "labels"
        if neu_labels.is_dir():
            return str(resolved)
        if (resolved / "labels").is_dir() and dataset_name.upper().replace("_", "-") in (
            "NEU-DET",
            "NEUDET",
        ):
            return str(resolved)

    expected = f"{dataset_name}/{label_dir_name}/"
    raise FileNotFoundError(f"dataset not found under data-root (expected {expected})")


def split_counts(n_total: int, val_size: float, test_size: float) -> tuple[int, int, int]:
    n_test = min(max(int(round(n_total * test_size)), 1), n_total - (MIN_SAMPLES_FOR_THREE_WAY_SPLIT - 1))
    n_val = min(max(int(round(n_total * val_size)), 1), n_total - n_test - 1)
    n_train = n_total - n_test - n_val
    return n_train, n_val, n_test


def split_indices(
    y: np.ndarray,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 44,
    tries: int = STRATIFIED_SPLIT_TRIES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n_total = len(y)
    if n_total < MIN_SAMPLES_FOR_THREE_WAY_SPLIT:
        raise ValueError(f"need at least {MIN_SAMPLES_FOR_THREE_WAY_SPLIT} samples")
    n_train, n_val, n_test = split_counts(n_total, val_size, test_size)
    global_mean = y.mean(axis=0)
    best_split = None
    best_gap = float("inf")
    best_valid = None
    best_valid_gap = float("inf")
    base_rng = np.random.default_rng(seed)
    for _ in range(tries):
        subseed = int(base_rng.integers(0, RANDOM_SUBSEED_MAX))
        rng = np.random.default_rng(subseed)
        perm = np.arange(n_total)
        rng.shuffle(perm)
        idx_test = perm[:n_test]
        idx_val = perm[n_test : n_test + n_val]
        idx_train = perm[n_test + n_val :]
        train_r = y[idx_train].mean(axis=0)
        val_r = y[idx_val].mean(axis=0)
        test_r = y[idx_test].mean(axis=0)
        gap = (
            float(np.mean(np.abs(train_r - global_mean)))
            + float(np.mean(np.abs(val_r - global_mean)))
            + float(np.mean(np.abs(test_r - global_mean)))
        )
        if gap < best_gap:
            best_gap = gap
            best_split = (idx_train, idx_val, idx_test)
        rare_ok = True
        for c in range(y.shape[1]):
            n_c = int(y[:, c].sum())
            if n_c < 3:
                continue
            if int(y[idx_train, c].sum()) < 1 or int(y[idx_val, c].sum()) < 1 or int(y[idx_test, c].sum()) < 1:
                rare_ok = False
                break
        if rare_ok and gap < best_valid_gap:
            best_valid_gap = gap
            best_valid = (idx_train, idx_val, idx_test)
    chosen = best_valid if best_valid is not None else best_split
    if chosen is None:
        raise RuntimeError("split failed")
    return chosen[0], chosen[1], chosen[2], best_valid is not None


def amp_autocast(use_amp: bool):
    if use_amp and RUN_DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()
