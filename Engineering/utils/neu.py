"""NEU-DET image-level single-label (6 classes; class parsed from filename prefix)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .core import Config

NEU_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NEU_NUM_CLASSES = len(NEU_CLASS_NAMES)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def apply_neu_preset(cfg: Config) -> None:
    cfg.dataset_name = "NEU-DET"
    cfg.label_dir_name = "labels"
    cfg.num_classes = NEU_NUM_CLASSES
    cfg.image_size = (200, 200)
    cfg.save_path = str(Path("checkpoints") / "neu_opt_8_16_32_dog_v2_best.pth")
    cfg.single_label = True
    cfg.use_per_class_temperature = False
    cfg.aux_weight = 0.08
    cfg.decouple_weight = 0.005
    cfg.decouple_warmup_epochs = 20


def _dataset_dir(cfg: Config) -> Path:
    root = Path(cfg.data_root)
    cand = root / cfg.dataset_name
    if (cand / "labels").is_dir():
        return cand
    if (root / "labels").is_dir():
        return root
    raise FileNotFoundError(f"NEU-DET/labels not found; checked: {cand / 'labels'}, {root / 'labels'}")


def _image_dirs(ds: Path) -> list[Path]:
    out: list[Path] = []
    for rel in ("images", "IMAGES", "train/images", "valid/images", "test/images"):
        p = ds / rel
        if p.is_dir():
            out.append(p)
    if not out and ds.is_dir():
        out.append(ds)
    return out


def parse_class_from_stem(stem: str) -> int:
    for i, name in enumerate(NEU_CLASS_NAMES):
        if stem.startswith(name + "_"):
            return i
    raise ValueError(f"cannot parse NEU class from filename: {stem}")


def _find_image(stem: str, search_dirs: list[Path], ds: Path) -> Path | None:
    for base in search_dirs:
        for ext in IMAGE_EXTS:
            p = base / f"{stem}{ext}"
            if p.is_file():
                return p
    for ext in IMAGE_EXTS:
        p = ds / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_neu_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    ds = _dataset_dir(cfg)
    labels_dir = ds / "labels"
    search_dirs = _image_dirs(ds)
    if not search_dirs:
        raise FileNotFoundError(f"NEU-DET image directory not found under {ds}; expected images/ or matching filenames.")

    images: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    missing_img = 0
    label_files = sorted(labels_dir.glob("*.txt"))
    for lp in tqdm(label_files, desc="Loading NEU-DET"):
        stem = lp.stem
        try:
            cls_id = parse_class_from_stem(stem)
        except ValueError:
            continue
        img_path = _find_image(stem, search_dirs, ds)
        if img_path is None:
            missing_img += 1
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            missing_img += 1
            continue
        gray = cv2.resize(gray, cfg.image_size).astype(np.float32) / 255.0
        y = np.zeros(cfg.num_classes, dtype=np.float32)
        y[cls_id] = 1.0
        images.append(gray)
        rows.append(y)

    if not images:
        raise ValueError(f"no valid NEU samples (labels {len(label_files)}, missing images {missing_img}).")
    if missing_img:
        print(f"warn: skipped {missing_img} labels without image", flush=True)
    return np.asarray(images, dtype=np.float32), np.asarray(rows, dtype=np.float32)
