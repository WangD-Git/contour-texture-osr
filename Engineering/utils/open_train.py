"""GC10 open-set recognition: DoG five-stage dual stream v2 + entropy/JS/prototype rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core
from .core import (
    FUSE_DEEP_CH,
    FUSE_SHALLOW_CH,
    AsymmetricLoss,
    Config,
    compose_dog_training_loss,
    eval_logits,
    multilabel_metrics,
    postprocess_multilabel_eval,
    predictions_from_probs,
)
from .model import dual_dog_scale_balance_loss
from .runtime import RUN_DEVICE, amp_autocast, split_indices
from .loop import train_multilabel_with_val

OSR_EXPERIMENTS: dict[str, int] = {
    "A": 10,  # Inclusion — highest main-head entropy AUROC across 10 classes
    "B": 1,  # Punching — highest JS disagreement AUROC (branch opposition)
    "C": 6,  # Pitted surface — lowest main-head entropy AUROC (semantic continuity)
    "D": 4,  # Patches — 2nd-lowest main-head entropy AUROC (paired with C)
}

# Experiments C + D both unknown: Patches(4) + Pitted surface(6) → 8-class closed training
OSR_CD_DUAL_UNKNOWN: tuple[int, int] = (4, 6)

GC10_UNKNOWN_NAMES: dict[int, str] = {
    1: "Punching",
    2: "Oil stain",
    3: "Silk spot",
    4: "Patches",
    5: "Foreign matter",
    6: "Pitted surface",
    7: "Scratches",
    8: "Rolled-in scale",
    9: "Zinc slag",
    10: "Inclusion",
}


@dataclass
class OSRConfig(Config):
    """OSR fields extending closed-set Config."""

    unknown_folder_id: int = 2  # single leave-one-unknown; compat field when unknown_folder_ids is set
    unknown_folder_ids: tuple[int, ...] | None = None  # multi-unknown, e.g. (4, 6)=Patches+Pitted surface
    use_prototype_loss: bool = False
    use_push_loss: bool = False
    use_aux_loss: bool = True
    use_decouple_loss: bool = True
    use_dog_prep: bool = True
    use_dog_bal_loss: bool = True
    proto_weight: float = 0.06
    push_weight: float = 0.02
    proto_ema: float = 0.99
    proto_warmup_epochs: int = 40
    proto_margin: float = 0.2
    push_margin: float = 0.15
    closed_init_ckpt: str = ""
    save_path: str = "checkpoints/osr_dog_v2_unknown2_best.pth"

    def effective_unknown_folder_ids(self) -> tuple[int, ...]:
        if self.unknown_folder_ids:
            return tuple(sorted({int(u) for u in self.unknown_folder_ids}))
        return (int(self.unknown_folder_id),)

    @property
    def unknown_label_indices(self) -> list[int]:
        return [int(u) - 1 for u in self.effective_unknown_folder_ids()]

    @property
    def known_class_indices(self) -> list[int]:
        unk = set(self.unknown_label_indices)
        return [i for i in range(10) if i not in unk]

    def apply_defaults(self) -> None:
        core.apply_preset(self)

    def apply_osr_model_dims(self) -> None:
        save_path = self.save_path
        self.num_classes = 10 - len(self.effective_unknown_folder_ids())
        self.apply_defaults()
        self.save_path = save_path


def labels_10_to_known(y10: np.ndarray, known_idx: list[int]) -> np.ndarray:
    return y10[:, known_idx].astype(np.float32)


def split_osr_dataset(x_all: np.ndarray, y_all: np.ndarray, cfg: OSRConfig) -> dict:
    if y_all.shape[1] != 10:
        raise ValueError(f"OSR requires 10-dim labels, got {y_all.shape}")
    unk_cols = cfg.unknown_label_indices
    known = cfg.known_class_indices
    has_unknown = np.any(y_all[:, unk_cols] > 0.5, axis=1)
    closed_mask = ~has_unknown

    x_closed = x_all[closed_mask]
    y_closed9 = labels_10_to_known(y_all[closed_mask], known)
    x_unknown = x_all[has_unknown]

    tr, va, te, _ = split_indices(
        y_closed9,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )
    return {
        "x_closed": x_closed,
        "y_closed9": y_closed9,
        "x_unknown": x_unknown,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
        "n_total": len(x_all),
        "n_closed_pool": len(x_closed),
        "n_unknown_pool": len(x_unknown),
        "unknown_name": _format_unknown_names(cfg),
        "unknown_folder_ids": list(cfg.effective_unknown_folder_ids()),
    }


def _format_unknown_names(cfg: OSRConfig) -> str:
    parts = [
        GC10_UNKNOWN_NAMES.get(uid, f"folder{uid}") for uid in cfg.effective_unknown_folder_ids()
    ]
    return " + ".join(parts)


def build_osr_loaders(split: dict, cfg: OSRConfig) -> dict:
    """Same make_loader as closed-set; train/val/test in closed pool + unknown pool."""
    x_c, y9 = split["x_closed"], split["y_closed9"]
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
    n_unk = len(split["x_unknown"])
    return {
        "train": core.make_loader(x_c[tr], y9[tr], cfg, train=True),
        "val": core.make_loader(x_c[va], y9[va], cfg, train=False),
        "test": core.make_loader(x_c[te], y9[te], cfg, train=False),
        "unknown": core.make_loader(
            split["x_unknown"],
            np.zeros((n_unk, cfg.num_classes), dtype=np.float32),
            cfg,
            train=False,
        ),
    }


class BranchPrototypes(nn.Module):
    """Per-branch class prototype bank (learnable + EMA update)."""

    def __init__(self, num_classes: int, dim: int, ema: float = 0.99):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.ema = float(ema)
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.register_buffer("_ema_init", torch.zeros(num_classes, dtype=torch.bool), persistent=False)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.prototypes.float(), dim=1)

    @torch.no_grad()
    def ema_update(self, feats: torch.Tensor, y: torch.Tensor) -> None:
        if feats.numel() == 0:
            return
        f = F.normalize(feats.detach().float(), dim=1)
        for c in range(self.num_classes):
            m = y[:, c] > 0.5
            if not m.any():
                continue
            mean_f = F.normalize(f[m].mean(dim=0), dim=0)
            if not bool(self._ema_init[c].item()):
                self.prototypes.data[c].copy_(mean_f)
                self._ema_init[c] = True
            else:
                self.prototypes.data[c].mul_(1.0 - self.ema).add_(mean_f, alpha=self.ema)


def build_prototype_banks(
    cfg: Config,
    *,
    proto_ema: float = 0.99,
) -> tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes]:
    branch_dim = FUSE_SHALLOW_CH + FUSE_DEEP_CH
    fusion_dim = FUSE_DEEP_CH
    n = cfg.num_classes
    return (
        BranchPrototypes(n, branch_dim, proto_ema).to(RUN_DEVICE),
        BranchPrototypes(n, branch_dim, proto_ema).to(RUN_DEVICE),
        BranchPrototypes(n, fusion_dim, proto_ema).to(RUN_DEVICE),
    )


def prototype_pull_loss(
    feats: torch.Tensor,
    y: torch.Tensor,
    bank: BranchPrototypes,
    margin: float,
) -> torch.Tensor:
    f = F.normalize(feats.float(), dim=1)
    p = bank().float()
    losses: list[torch.Tensor] = []
    for c in range(y.shape[1]):
        m = y[:, c] > 0.5
        if not m.any():
            continue
        sim = (f[m] * p[c]).sum(dim=1)
        losses.append(F.relu(1.0 - float(margin) - sim).mean())
    if not losses:
        return feats.new_zeros(())
    return torch.stack(losses).mean()


def prototype_push_loss(bank: BranchPrototypes, margin: float) -> torch.Tensor:
    p = bank().float()
    sim = p @ p.t()
    n = p.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=p.device)
    return F.relu(sim[mask] - (1.0 - float(margin))).mean()


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp(eps, 1.0 - eps)
    q = q.float().clamp(eps, 1.0 - eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=1)
    kl_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def multilabel_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.float().clamp(eps, 1.0 - eps)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return h.mean(dim=1)


@torch.no_grad()
def hesitation_entropy(
    logits_main: torch.Tensor,
    logits_contour: torch.Tensor,
    logits_texture: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (main-head entropy, contour aux entropy, texture aux entropy, fused max of three)."""
    pm = torch.sigmoid(logits_main.float())
    pc = torch.sigmoid(logits_contour.float())
    pt = torch.sigmoid(logits_texture.float())
    h_main = multilabel_entropy(pm)
    h_contour = multilabel_entropy(pc)
    h_texture = multilabel_entropy(pt)
    h_fuse = torch.maximum(h_main, torch.maximum(h_contour, h_texture))
    return h_main, h_contour, h_texture, h_fuse


@torch.no_grad()
def dual_stream_disagreement(aux_c: torch.Tensor, aux_t: torch.Tensor) -> torch.Tensor:
    pc = torch.sigmoid(aux_c.float())
    pt = torch.sigmoid(aux_t.float())
    return _js_divergence(pc, pt)


def min_proto_distance_score(
    feat_c: torch.Tensor,
    feat_t: torch.Tensor,
    feat_f: torch.Tensor,
    bank_c: BranchPrototypes,
    bank_t: BranchPrototypes,
    bank_f: BranchPrototypes,
) -> torch.Tensor:
    def _min_dist(f: torch.Tensor, bank: BranchPrototypes) -> torch.Tensor:
        fn = F.normalize(f.float(), dim=1)
        p = bank().float()
        sim = fn @ p.t()
        return 1.0 - sim.max(dim=1).values

    return (_min_dist(feat_c, bank_c) + _min_dist(feat_t, bank_t) + _min_dist(feat_f, bank_f)) / 3.0


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int64).reshape(-1)
    s = scores.astype(np.float64).reshape(-1)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = 0.0
    for ps in pos:
        count += float((neg < ps).sum()) + 0.5 * float((neg == ps).sum())
    return count / (len(pos) * len(neg))


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _snapshot_banks(
    banks: tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes],
) -> tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes]:
    copies = tuple(BranchPrototypes(b.num_classes, b.dim, b.ema).to(RUN_DEVICE) for b in banks)
    for nb, ob in zip(copies, banks):
        nb.load_state_dict(ob.state_dict())
    return copies


@torch.no_grad()
def mean_branch_cosine_similarity(model: nn.Module, loader, cfg: OSRConfig) -> float:
    """Mean |cos| of contour/texture L1 features on closed-set eval (matches L_dec)."""
    model.eval()
    chunks: list[np.ndarray] = []
    for x, _ in loader:
        x = x.to(RUN_DEVICE, non_blocking=True)
        if cfg.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with amp_autocast(cfg.use_amp):
            out = model(x)
        fc = F.normalize(out["feat_contour_l1"].float(), dim=1)
        ft = F.normalize(out["feat_texture_l1"].float(), dim=1)
        chunks.append(torch.abs((fc * ft).sum(dim=1)).cpu().numpy())
    if not chunks:
        return float("nan")
    return float(np.concatenate(chunks, axis=0).mean())


@torch.no_grad()
def collect_osr_scores(
    model: nn.Module,
    loader,
    cfg: OSRConfig,
    banks: tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes] | None,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "disagreement": [],
        "entropy": [],
        "entropy_main": [],
        "entropy_contour": [],
        "entropy_texture": [],
        "max_prob": [],
    }
    use_proto = banks is not None
    if use_proto:
        chunks["proto_dist"] = []

    for x, _ in loader:
        x = x.to(RUN_DEVICE, non_blocking=True)
        if cfg.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with amp_autocast(cfg.use_amp):
            out = model(x)
        lm = out["logits"].float()
        lc = out["aux_contour"].float()
        lt = out["aux_texture"].float()
        pm = torch.sigmoid(lm)
        h_main, h_contour, h_texture, h_fuse = hesitation_entropy(lm, lc, lt)
        chunks["disagreement"].append(dual_stream_disagreement(lc, lt).cpu().numpy())
        chunks["entropy"].append(h_fuse.cpu().numpy())
        chunks["entropy_main"].append(h_main.cpu().numpy())
        chunks["entropy_contour"].append(h_contour.cpu().numpy())
        chunks["entropy_texture"].append(h_texture.cpu().numpy())
        chunks["max_prob"].append((1.0 - pm.max(dim=1).values).cpu().numpy())
        if use_proto:
            pd = min_proto_distance_score(
                out["feat_contour_l1"],
                out["feat_texture_l1"],
                out["feat_fusion"],
                banks[0],
                banks[1],
                banks[2],
            )
            chunks["proto_dist"].append(pd.cpu().numpy())

    out_np = {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}
    dis = out_np["disagreement"]
    ent = out_np["entropy"]
    mp = out_np["max_prob"]
    out_np["combined"] = 0.4 * _zscore(dis) + 0.4 * _zscore(ent) + 0.2 * _zscore(mp)
    return out_np


@torch.no_grad()
def evaluate_osr(
    model: nn.Module,
    closed_test_loader,
    unknown_loader,
    cfg: OSRConfig,
    banks: tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes] | None,
    *,
    calib_loader=None,
) -> dict[str, float]:
    """Closed-set μF1 and open-set AUROC; calib_loader fits T/τ on Val (train.py protocol)."""
    if calib_loader is not None:
        cal_logits, cal_y = eval_logits(model, calib_loader, cfg, use_tta=True)
        te_logits, te_y = eval_logits(model, closed_test_loader, cfg, use_tta=True)
        _, _, _, closed_m_dict = postprocess_multilabel_eval(cal_logits, cal_y, te_logits, te_y, cfg)
        closed_m = closed_m_dict
    else:
        closed_logits, closed_y = eval_logits(model, closed_test_loader, cfg, use_tta=True)
        temps = core.fit_multilabel_calibration(closed_logits, closed_y, cfg)
        probs = core.logits_to_probs(closed_logits, temps)
        taus = core.tune_per_class_thresholds(probs, closed_y, cfg)
        closed_m = multilabel_metrics(closed_y, predictions_from_probs(probs, cfg, taus))

    s_closed = collect_osr_scores(model, closed_test_loader, cfg, banks)
    s_unknown = collect_osr_scores(model, unknown_loader, cfg, banks)
    y_bin = np.concatenate(
        [np.zeros(len(s_closed["entropy"]), dtype=np.float32), np.ones(len(s_unknown["entropy"]), dtype=np.float32)]
    )

    metrics: dict[str, float] = {
        "closed_micro_f1": float(closed_m["micro_f1"]),
        "feature_cosine_mean": mean_branch_cosine_similarity(model, closed_test_loader, cfg),
        "n_closed_eval": float(len(s_closed["entropy"])),
        "n_unknown_eval": float(len(s_unknown["entropy"])),
    }

    for key in (
        "disagreement",
        "entropy",
        "entropy_main",
        "entropy_contour",
        "entropy_texture",
        "max_prob",
        "combined",
        "proto_dist",
    ):
        c_arr = s_closed.get(key)
        u_arr = s_unknown.get(key)
        if c_arr is None or u_arr is None:
            metrics[f"auroc_{key}"] = float("nan")
            continue
        scores = np.concatenate([c_arr, u_arr])
        metrics[f"auroc_{key}"] = binary_auroc(y_bin, scores)
        metrics[f"{key}_closed_mean"] = float(c_arr.mean())
        metrics[f"{key}_unknown_mean"] = float(u_arr.mean())
        metrics[f"{key}_gap_mean"] = float(u_arr.mean() - c_arr.mean())

    return metrics


def train_osr_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: OSRConfig,
) -> tuple[float, tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes] | None]:
    """Shared compose_dog_training_loss + train_loop with closed-set."""
    crit = AsymmetricLoss(cfg.asl_gamma_neg, cfg.asl_gamma_pos, cfg.asl_clip)
    banks = build_prototype_banks(cfg, proto_ema=cfg.proto_ema) if cfg.use_prototype_loss else None
    w_aux = float(np.clip(cfg.aux_contour_weight, 0.05, 0.95))
    w_bal = float(cfg.dog_scale_bal_weight)
    best_banks_snap: tuple[BranchPrototypes, BranchPrototypes, BranchPrototypes] | None = None

    def loss_fn(m: nn.Module, out: dict, y: torch.Tensor, epoch: int) -> torch.Tensor:
        return compose_dog_training_loss(
            out,
            y,
            m,
            cfg,
            crit,
            epoch,
            use_aux=cfg.use_aux_loss,
            use_decouple=cfg.use_decouple_loss,
            use_dog_bal=cfg.use_dog_bal_loss,
            w_aux=w_aux,
            w_bal=w_bal,
            device=RUN_DEVICE,
            dog_bal_fn=dual_dog_scale_balance_loss,
        )

    def after_loss_fn(
        _m: nn.Module,
        out: dict,
        y: torch.Tensor,
        loss: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        if banks is None or epoch < cfg.proto_warmup_epochs:
            return loss
        proto_on = cfg.use_prototype_loss
        push_on = cfg.use_push_loss
        if proto_on:
            for bank, feats in zip(
                banks,
                (out["feat_contour_l1"], out["feat_texture_l1"], out["feat_fusion"]),
            ):
                bank.ema_update(feats.detach(), y)
            lp = (
                prototype_pull_loss(out["feat_contour_l1"], y, banks[0], cfg.proto_margin)
                + prototype_pull_loss(out["feat_texture_l1"], y, banks[1], cfg.proto_margin)
                + prototype_pull_loss(out["feat_fusion"], y, banks[2], cfg.proto_margin)
            ) / 3.0
            loss = loss + float(cfg.proto_weight) * lp
        if push_on:
            lp_push = (
                prototype_push_loss(banks[0], cfg.push_margin)
                + prototype_push_loss(banks[1], cfg.push_margin)
                + prototype_push_loss(banks[2], cfg.push_margin)
            ) / 3.0
            loss = loss + float(cfg.push_weight) * lp_push
        return loss

    def on_best(_f1: float, _ep: int) -> None:
        nonlocal best_banks_snap
        if banks is not None:
            best_banks_snap = _snapshot_banks(banks)

    def on_epoch_end(epoch: int, vm: dict, best_f1: float, best_epoch: int) -> None:
        if epoch % cfg.log_interval != 0:
            return
        proto_on = cfg.use_prototype_loss and banks is not None and epoch >= cfg.proto_warmup_epochs
        flag = "proto=on" if proto_on else "proto=off"
        print(
            f"Ep{epoch:03d} | closed_val_μF1={vm['micro_f1']:.4f} | "
            f"best={best_f1:.4f} @ep{best_epoch} | {flag}",
            flush=True,
        )

    best_f1 = train_multilabel_with_val(
        model,
        train_loader,
        val_loader,
        cfg,
        loss_fn,
        warmup_epochs=0,
        after_loss_fn=after_loss_fn if banks is not None else None,
        extra_modules=list(banks) if banks is not None else None,
        on_best=on_best if banks is not None else None,
        on_epoch_end=on_epoch_end,
        verbose=False,
    )
    if best_banks_snap is not None and banks is not None:
        for bank, bb in zip(banks, best_banks_snap):
            bank.load_state_dict(bb.state_dict())
    return float(best_f1), banks
