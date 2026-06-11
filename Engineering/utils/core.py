"""Learnable DoG contour-texture dual stream (8→16→32) + GC10/NEU data and training utilities."""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .runtime import (
    DEVICE,
    RUN_DEVICE,
    amp_autocast,
    move_model_with_fallback,
    resolve_data_root,
    set_run_device,
    set_seed,
    split_indices,
    torch_load_checkpoint,
)

DIV_EPS = 1e-6
GRAY_NORM_EPS = 1e-6
GAUSSIAN_BLUR_KERNEL_SIZE = 9
GAUSSIAN_BLUR_SIGMA = 2.0
XML_SUFFIX = ".xml"
DATASET_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

AUG_FLIP_PROB = 0.5
AUG_INTENSITY_PROB = 0.3
AUG_SCALE_LOW, AUG_SCALE_HIGH = 0.85, 1.15
AUG_BIAS_LOW, AUG_BIAS_HIGH = -0.04, 0.04


@dataclass
class Config:
    data_root: str = "./"
    dataset_name: str = "GC10-DET"
    label_dir_name: str = "lable"
    num_classes: int = 10
    image_size: tuple[int, int] = (192, 192)
    batch_size: int = 8
    epochs: int = 100
    val_size: float = 0.2
    test_size: float = 0.2
    lr: float = 5e-4
    weight_decay: float = 1e-4
    cosine_eta_min: float = 1e-4
    num_workers: int = 2
    log_interval: int = 10
    seed: int = 44
    channel_widths: tuple[int, int, int] | None = (8, 16, 32)
    use_eca: bool = True
    use_blur_pool: bool = True
    blur_pool_filt_size: int = 3
    use_amp: bool = True
    use_channels_last: bool = False
    use_adaptive_loss_balance: bool = False
    sync_h2d: bool = False
    use_tta_eval: bool = True
    multilabel_threshold: float = 0.5
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    threshold_grid_min: float = 0.2
    threshold_grid_max: float = 0.8
    threshold_grid_step: float = 0.02
    val_threshold_tune_interval: int = 2
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    aux_weight: float = 0.10
    decouple_weight: float = 0.01
    decouple_warmup_epochs: int = 40
    aux_contour_weight: float = 0.5
    use_per_class_temperature: bool = True
    temperature_grid_min: float = 0.15
    temperature_grid_max: float = 4.0
    temperature_grid_step: float = 0.1
    calibrate_during_train: bool = False
    single_label: bool = False
    dog_scale_bal_weight: float = 0.001
    save_path: str = "checkpoints/ification_opt_8_16_32_blurpool_dog_v2_best.pth"

    @property
    def num_defect_classes(self) -> int:
        return self.num_classes


OPT_CHANNEL_WIDTHS: tuple[int, int, int] = (8, 16, 32)
STREAM_WIDTHS = OPT_CHANNEL_WIDTHS
FUSE_SHALLOW_CH = 8   # shallow S (stage 2, 8ch)
FUSE_DEEP_CH = 32     # deep D (stage 5, 32ch)
HEAD_DIM = FUSE_DEEP_CH


def closed_ckpt_stem(cfg: Config) -> str:
    tag = "_blurpool" if cfg.use_blur_pool else ""
    return f"ification_opt_8_16_32{tag}_dog_v2"


def osr_ckpt_stem(_cfg: Config) -> str:
    return "dog_v2"


def apply_preset(cfg: Config) -> None:
    """Five-stage dual stream 8→16→32; learnable DoG + high-pass texture frontend."""
    cfg.channel_widths = OPT_CHANNEL_WIDTHS
    cfg.use_eca = True
    cfg.aux_weight = 0.10
    cfg.decouple_weight = 0.01
    if not cfg.single_label:
        cfg.save_path = f"checkpoints/{closed_ckpt_stem(cfg)}_best.pth"
        # Match exp.py baseline: val μF1 @τ=0.5 no TTA for model selection + 30-epoch early stop
        if cfg.early_stop_patience <= 0:
            cfg.early_stop_patience = 30
        # Baselines use channels_last=False; True makes CUDA validation often non-deterministic
        cfg.use_channels_last = False


def infer_use_blur_pool_from_state_dict(state: dict[str, Any]) -> bool:
    """BlurPool if ckpt has *.pool.weight, else MaxPool."""
    sd = state.get("model_state_dict", state)
    return any(str(k).endswith(".pool.weight") for k in sd.keys())


def apply_pool_backend_from_ckpt(cfg: Config, state: dict[str, Any]) -> bool:
    """Align downsampling module before loading weights (BlurPool vs MaxPool)."""
    use_blur = infer_use_blur_pool_from_state_dict(state)
    cfg.use_blur_pool = use_blur
    return use_blur


def pool_backend_label(use_blur_pool: bool) -> str:
    return "BlurPool" if use_blur_pool else "MaxPool"


def parse_class_id_from_name(name: str) -> int:
    prefix = name.strip().split("_", 1)[0]
    return int(prefix) - 1


def build_image_index(dataset_root: str, label_dir_name: str) -> dict[str, str]:
    """Filename → path; sorted folders/files for reproducible runs."""
    out: dict[str, str] = {}
    for folder in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path) or folder == label_dir_name:
            continue
        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() in DATASET_IMAGE_EXTENSIONS and fname not in out:
                out[fname] = os.path.join(folder_path, fname)
    return out


def labels_from_xml(root: ET.Element, num_classes: int) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        try:
            c = parse_class_id_from_name(name)
        except ValueError:
            continue
        if 0 <= c < num_classes:
            y[c] = 1.0
    return y


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = (size - 1) / 2.0
    ax = torch.arange(size, device=device, dtype=dtype) - r
    g = torch.exp(-(ax.view(1, -1) ** 2 + ax.view(-1, 1) ** 2) / (2 * sigma**2))
    return (g / g.sum()).view(1, 1, size, size)


def load_multilabel_dataset(cfg: Config, image_index: dict[str, str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = os.path.join(cfg.data_root, cfg.dataset_name)
    label_dir = os.path.join(root, cfg.label_dir_name)
    if image_index is None:
        image_index = build_image_index(root, cfg.label_dir_name)
    images, rows = [], []
    for xml_name in tqdm(sorted(os.listdir(label_dir)), desc="Loading GC10-DET"):
        if not xml_name.lower().endswith(XML_SUFFIX):
            continue
        doc = ET.parse(os.path.join(label_dir, xml_name)).getroot()
        fname = doc.findtext("filename")
        if not fname or fname.strip() not in image_index:
            continue
        y = labels_from_xml(doc, cfg.num_classes)
        if y.sum() == 0:
            continue
        gray = cv2.imread(image_index[fname.strip()], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = cv2.resize(gray, cfg.image_size).astype(np.float32) / 255.0
        images.append(gray)
        rows.append(y)
    if not images:
        raise ValueError("no valid samples")
    return np.asarray(images, dtype=np.float32), np.asarray(rows, dtype=np.float32)


def prepare_full_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    cfg.data_root = resolve_data_root(
        cfg.data_root if str(cfg.data_root).strip() else None,
        dataset_name=cfg.dataset_name,
        label_dir_name=cfg.label_dir_name,
    )
    if cfg.dataset_name.upper().replace("_", "-") in ("NEU-DET", "NEUDET"):
        from .neu import load_neu_dataset

        return load_neu_dataset(cfg)
    return load_multilabel_dataset(cfg)


def _augment_gray(gray_1hw: torch.Tensor) -> torch.Tensor:
    x = gray_1hw
    if random.random() < AUG_FLIP_PROB:
        x = torch.flip(x, [2])
    if random.random() < AUG_INTENSITY_PROB:
        x = torch.clamp(x * random.uniform(AUG_SCALE_LOW, AUG_SCALE_HIGH) + random.uniform(AUG_BIAS_LOW, AUG_BIAS_HIGH), 0, 1)
    return x


class MultiLabelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool, cfg: Config):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.train = train
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.x[i].unsqueeze(0)
        if self.train:
            g = _augment_gray(g)
        return g, self.y[i]


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed DataLoader worker RNG."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(x: np.ndarray, y: np.ndarray, cfg: Config, train: bool) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed))
    nw = int(cfg.num_workers)
    return DataLoader(
        MultiLabelDataset(x, y, train, cfg),
        batch_size=cfg.batch_size,
        shuffle=train,
        generator=gen if train else None,
        num_workers=nw,
        worker_init_fn=_seed_dataloader_worker if nw > 0 else None,
        pin_memory=RUN_DEVICE.type == "cuda",
        persistent_workers=nw > 0,
    )


class ECA(nn.Module):
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        pad = (k - 1) // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv1d(1, 1, k, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.act(self.fc(self.pool(x).squeeze(-1).transpose(1, 2))).transpose(1, 2).unsqueeze(-1)
        return x * w


def _binomial_blur_kernel(filt_size: int) -> torch.Tensor:
    if filt_size == 3:
        v = torch.tensor([1.0, 2.0, 1.0])
    elif filt_size == 5:
        v = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    else:
        raise ValueError(f"BlurPool filt_size must be 3 or 5, got {filt_size}")
    k = v[:, None] * v[None, :]
    return k / k.sum()


class BlurPool2d(nn.Module):
    """Depthwise binomial low-pass + stride=2 downsampling (anti-aliasing MaxPool substitute)."""

    def __init__(self, channels: int, *, filt_size: int = 3, stride: int = 2):
        super().__init__()
        if stride != 2:
            raise ValueError("BlurPool2d currently supports stride=2 only")
        k = _binomial_blur_kernel(int(filt_size))
        weight = k.view(1, 1, k.shape[0], k.shape[1]).repeat(int(channels), 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = int(channels)
        self.stride = int(stride)
        self.pad = (k.shape[0] - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.pad,
            groups=self.channels,
        )


def _make_stride2_pool(
    channels: int,
    *,
    use_blur_pool: bool,
    filt_size: int = 3,
) -> nn.Module:
    if use_blur_pool:
        return BlurPool2d(channels, filt_size=filt_size)
    return nn.MaxPool2d(2, 2)


class ConvPoolDown(nn.Module):
    """stride=1 conv + BlurPool/MaxPool(2)."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        padding: int,
        *,
        use_blur_pool: bool = True,
        blur_pool_filt_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, int(kernel), stride=1, padding=int(padding), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=False),
        )
        self.pool = _make_stride2_pool(
            c_out,
            use_blur_pool=use_blur_pool,
            filt_size=blur_pool_filt_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class ConvEnhance(nn.Module):
    """Same-scale enhancement conv (stride=1)."""

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        pad = int(kernel) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, int(kernel), stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DSConvPoolBlock(nn.Module):
    """DWConv(stride=1) → PWConv → ECA → BlurPool/MaxPool."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        use_eca: bool = True,
        downsample: bool = True,
        kernel: int = 3,
        *,
        use_blur_pool: bool = True,
        blur_pool_filt_size: int = 3,
    ):
        super().__init__()
        k = int(kernel)
        pad = k // 2
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_in, k, stride=1, padding=pad, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU6(inplace=False),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=False),
        )
        self.eca = ECA(c_out) if use_eca else nn.Identity()
        self.pool = (
            _make_stride2_pool(c_out, use_blur_pool=use_blur_pool, filt_size=blur_pool_filt_size)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.eca(self.body(x)))


class FiveStageStream(nn.Module):
    """v2 five-stage 8→16→32 backbone (192 input)."""

    def __init__(
        self,
        kernel: int,
        *,
        in_ch: int = 1,
        widths: tuple[int, int, int] = OPT_CHANNEL_WIDTHS,
        use_eca: bool = True,
        use_blur_pool: bool = True,
        blur_pool_filt_size: int = 3,
    ):
        super().__init__()
        c8, c16, c32 = (int(widths[0]), int(widths[1]), int(widths[2]))
        k = int(kernel)
        pad = k // 2
        pool_kw = dict(use_blur_pool=use_blur_pool, blur_pool_filt_size=blur_pool_filt_size)
        self.stem = ConvPoolDown(int(in_ch), c8, kernel=k, padding=pad, **pool_kw)
        self.enh_shallow = ConvEnhance(c8, c8, k)
        self.to16 = DSConvPoolBlock(c8, c16, use_eca=use_eca, downsample=True, kernel=k, **pool_kw)
        self.to32 = DSConvPoolBlock(c16, c32, use_eca=use_eca, downsample=True, kernel=k, **pool_kw)
        self.enh_deep = ConvEnhance(c32, c32, k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        shallow = self.enh_shallow(x)
        x = self.to16(shallow)
        x = self.to32(x)
        deep = self.enh_deep(x)
        return shallow, deep


class Concat1x1Fusion(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU6(inplace=False),
        )

    def forward(self, fc: torch.Tensor, ft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.mix(torch.cat([fc, ft], dim=1))
        return fused, fc, ft


class DualStreamNet(nn.Module):
    """v2 dual stream: main head cross-branch GAP(8)+GAP(32); aux heads branch-internal 8+32."""

    def __init__(
        self,
        cfg: Config,
        *,
        contour_kernel: int = 5,
        texture_kernel: int = 3,
    ):
        super().__init__()
        c8, _, c32 = OPT_CHANNEL_WIDTHS
        self.widths = OPT_CHANNEL_WIDTHS
        stream_kw = dict(
            widths=OPT_CHANNEL_WIDTHS,
            use_eca=cfg.use_eca,
            use_blur_pool=cfg.use_blur_pool,
            blur_pool_filt_size=cfg.blur_pool_filt_size,
        )
        self.contour = FiveStageStream(int(contour_kernel), **stream_kw)
        self.texture = FiveStageStream(int(texture_kernel), **stream_kw)
        self.cross_fuse8 = Concat1x1Fusion(c8)
        self.cross_fuse32 = Concat1x1Fusion(c32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        fuse_dim = c8 + c32
        self.proj = nn.Sequential(
            nn.Linear(fuse_dim, c32, bias=False),
            nn.LayerNorm(c32),
            nn.ReLU6(inplace=False),
        )
        self.head = nn.Linear(c32, cfg.num_classes)
        self.aux_contour = nn.Linear(fuse_dim, cfg.num_classes)
        self.aux_texture = nn.Linear(fuse_dim, cfg.num_classes)

    def _forward_streams(
        self,
        x_contour: torch.Tensor,
        x_texture: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sc, dc = self.contour(x_contour)
        st, dt = self.texture(x_texture)
        f8, _, _ = self.cross_fuse8(sc, st)
        f32, _, _ = self.cross_fuse32(dc, dt)
        pf8 = self.pool(f8).flatten(1)
        pf32 = self.pool(f32).flatten(1)
        feat_fusion = self.proj(torch.cat([pf8, pf32], dim=1))
        feat_contour = torch.cat([self.pool(sc).flatten(1), self.pool(dc).flatten(1)], dim=1)
        feat_texture = torch.cat([self.pool(st).flatten(1), self.pool(dt).flatten(1)], dim=1)
        return {
            "logits": self.head(feat_fusion),
            "aux_contour": self.aux_contour(feat_contour),
            "aux_texture": self.aux_texture(feat_texture),
            "feat_contour_l1": feat_contour,
            "feat_texture_l1": feat_texture,
            "feat_fusion": feat_fusion,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._forward_streams(x[:, :1], x[:, 1:2])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dual_stream_net(
    cfg: Config,
    *,
    contour_kernel: int = 5,
    texture_kernel: int = 3,
) -> DualStreamNet:
    w = cfg.channel_widths or OPT_CHANNEL_WIDTHS
    if tuple(int(x) for x in w) != OPT_CHANNEL_WIDTHS:
        raise ValueError(f"dual stream supports channels {OPT_CHANNEL_WIDTHS} only, got {w}")
    return DualStreamNet(
        cfg,
        contour_kernel=contour_kernel,
        texture_kernel=texture_kernel,
    )


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gn, self.gp, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        if self.clip > 0:
            p = (p + self.clip).clamp(max=1.0)
        loss_pos = target * torch.log(p.clamp(min=1e-8))
        loss_neg = (1.0 - target) * torch.log((1.0 - p).clamp(min=1e-8))
        pt = p * target + (1.0 - p) * (1.0 - target)
        w = (1.0 - pt).pow(self.gp) * target + (1.0 - pt).pow(self.gn) * (1.0 - target)
        return -(w * (loss_pos + loss_neg)).mean()


def decouple_loss(fc: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    c = F.normalize(fc, p=2, dim=1)
    t = F.normalize(ft, p=2, dim=1)
    return torch.mean(torch.abs((c * t).sum(dim=1)))


def adaptive_loss_weights(main: torch.Tensor, aux: torch.Tensor, dec: torch.Tensor, cfg: Config, epoch: int) -> tuple[float, float]:
    warm = min(1.0, epoch / max(1, cfg.decouple_warmup_epochs))
    if not cfg.use_adaptive_loss_balance:
        return cfg.aux_weight * warm, cfg.decouple_weight * warm
    # Per-batch dynamic scaling can jitter weights across GPU runs (baselines omit this path)
    ra = float((main.detach() / (aux.detach() + DIV_EPS)).clamp(0.5, 2.0).item())
    rd = float((main.detach() / (dec.detach() + DIV_EPS)).clamp(0.5, 2.0).item())
    return cfg.aux_weight * warm * ra, cfg.decouple_weight * warm * rd


def compose_dog_training_loss(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    model: nn.Module,
    cfg: Config,
    crit: nn.Module,
    epoch: int,
    *,
    use_aux: bool = True,
    use_decouple: bool = True,
    use_dog_bal: bool = True,
    w_aux: float | None = None,
    w_bal: float | None = None,
    device: torch.device | None = None,
    dog_bal_fn=None,
) -> torch.Tensor:
    """Unified DoG v2 closed-set/OSR training loss."""
    if w_aux is None:
        w_aux = float(np.clip(cfg.aux_contour_weight, 0.05, 0.95))
    if w_bal is None:
        w_bal = float(cfg.dog_scale_bal_weight)
    if device is None:
        device = RUN_DEVICE
    if dog_bal_fn is None:
        from .model import dual_dog_scale_balance_loss

        dog_bal_fn = dual_dog_scale_balance_loss

    lm = crit(out["logits"], y)
    loss = lm
    la = (
        w_aux * crit(out["aux_contour"], y) + (1.0 - w_aux) * crit(out["aux_texture"], y)
        if use_aux
        else None
    )
    ld = (
        decouple_loss(out["feat_contour_l1"], out["feat_texture_l1"])
        if use_decouple
        else None
    )
    _eps = 1e-6
    wa, wd = adaptive_loss_weights(
        lm,
        la if la is not None else lm.detach().mul(0).add_(_eps),
        ld if ld is not None else lm.detach().mul(0).add_(_eps),
        cfg,
        epoch,
    )
    if la is not None:
        loss = loss + wa * la
    if ld is not None:
        loss = loss + wd * ld
    if use_dog_bal:
        loss = loss + dog_bal_fn(model, w_bal, device=device)
    return loss


def _f1(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float((y_true * y_pred).sum())
    fp = float((y_pred * (1 - y_true)).sum())
    fn = float((y_true * (1 - y_pred)).sum())
    return {"micro_f1": _f1(tp, fp, fn), "micro_precision": tp / (tp + fp + 1e-8), "micro_recall": tp / (tp + fn + 1e-8)}


def tune_per_class_thresholds(probs: np.ndarray, y_true: np.ndarray, cfg: Config) -> np.ndarray:
    grid = np.arange(cfg.threshold_grid_min, cfg.threshold_grid_max + 1e-6, cfg.threshold_grid_step)
    taus = np.full((y_true.shape[1],), cfg.multilabel_threshold, dtype=np.float32)
    for c in range(y_true.shape[1]):
        best_f1, best_t = -1.0, cfg.multilabel_threshold
        for t in grid:
            pred = (probs[:, c] >= t).astype(np.float32)
            f1 = _f1(float((y_true[:, c] * pred).sum()), float((pred * (1 - y_true[:, c])).sum()), float((y_true[:, c] * (1 - pred)).sum()))
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        taus[c] = best_t if best_f1 >= 0 else cfg.multilabel_threshold
    # If all predictions are zero (μF1=0), fall back to default threshold
    if threshold_predict(probs, taus).sum() == 0:
        taus = np.full((y_true.shape[1],), cfg.multilabel_threshold, dtype=np.float32)
    return taus


def fit_scalar_temperature(logits: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    grid = np.arange(cfg.temperature_grid_min, cfg.temperature_grid_max + 1e-6, cfg.temperature_grid_step)
    z, yy = logits.astype(np.float64), y.astype(np.float64)
    best_t, best_loss = 1.0, float("inf")
    for T in grid:
        p = 1.0 / (1.0 + np.exp(-np.clip(z / max(T, 1e-3), -60, 60)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        if loss < best_loss:
            best_loss, best_t = loss, float(T)
    return np.full((y.shape[1],), best_t, dtype=np.float32)


def logits_to_probs(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) / temperatures.reshape(1, -1).astype(np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))).astype(np.float32)


def threshold_predict(probs: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return (probs >= taus.reshape(1, -1)).astype(np.float32)


def predictions_from_probs(probs: np.ndarray, cfg: Config, taus: np.ndarray | None = None) -> np.ndarray:
    if taus is None:
        taus = np.full((probs.shape[1],), cfg.multilabel_threshold, dtype=np.float32)
    return threshold_predict(probs, taus)


def should_tune_thresholds(epoch: int, cfg: Config) -> bool:
    interval = max(1, int(cfg.val_threshold_tune_interval))
    return epoch == 1 or epoch == cfg.epochs or epoch % interval == 0


def fit_multilabel_calibration(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if cfg.use_per_class_temperature:
        return fit_scalar_temperature(val_logits, val_y, cfg)
    return np.full((cfg.num_classes,), 1.0, dtype=np.float32)


def postprocess_multilabel_eval(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    """Fit T/τ on Val, apply on Test (closed-set final eval protocol)."""
    temps = fit_multilabel_calibration(val_logits, val_y, cfg)
    val_probs = logits_to_probs(val_logits, temps)
    test_probs = logits_to_probs(test_logits, temps)
    taus = tune_per_class_thresholds(val_probs, val_y, cfg)
    val_m = multilabel_metrics(val_y, predictions_from_probs(val_probs, cfg, taus))
    test_m = multilabel_metrics(test_y, predictions_from_probs(test_probs, cfg, taus))
    return temps, taus, val_m, test_m


def multilabel_val_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    *,
    taus: np.ndarray,
    temps: np.ndarray,
    epoch: int,
    use_tta: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """In-loop validation: fixed τ=0.5 by default; fit T/τ when calibrate_during_train=True."""
    v_logits, v_y = eval_logits(model, val_loader, cfg, use_tta=use_tta)

    if not cfg.calibrate_during_train:
        probs = logits_to_probs(v_logits, np.ones((cfg.num_classes,), dtype=np.float32))
        taus = np.full((cfg.num_classes,), cfg.multilabel_threshold, dtype=np.float32)
        metrics = multilabel_metrics(v_y, predictions_from_probs(probs, cfg, taus))
        return metrics, temps, taus

    temps = fit_multilabel_calibration(v_logits, v_y, cfg)
    probs = logits_to_probs(v_logits, temps)
    default_taus = np.full((cfg.num_classes,), cfg.multilabel_threshold, dtype=np.float32)
    default_m = multilabel_metrics(v_y, predictions_from_probs(probs, cfg, default_taus))

    if should_tune_thresholds(epoch, cfg):
        candidate = tune_per_class_thresholds(probs, v_y, cfg)
        tuned_m = multilabel_metrics(v_y, predictions_from_probs(probs, cfg, candidate))
        if tuned_m["micro_recall"] > 0 and tuned_m["micro_f1"] >= default_m["micro_f1"] - 0.05:
            taus = candidate
            metrics = tuned_m
        else:
            taus = default_taus
            metrics = default_m
    else:
        metrics = multilabel_metrics(v_y, predictions_from_probs(probs, cfg, taus))
        if metrics["micro_recall"] == 0 and default_m["micro_f1"] > 0:
            taus = default_taus
            metrics = default_m

    return metrics, temps, taus


@torch.no_grad()
def eval_logits(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    *,
    use_tta: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    do_tta = cfg.use_tta_eval if use_tta is None else use_tta
    logits_l, labels_l = [], []
    for x, y in loader:
        x = x.to(RUN_DEVICE, non_blocking=RUN_DEVICE.type == "cuda" and not cfg.sync_h2d)
        if cfg.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with amp_autocast(cfg.use_amp):
            out = model(x)
            lm = out["logits"]
            if do_tta:
                lm = 0.5 * (lm + model(torch.flip(x, dims=[3]))["logits"])
        logits_l.append(lm.cpu().numpy())
        labels_l.append(y.numpy())
    return np.concatenate(logits_l), np.concatenate(labels_l)
