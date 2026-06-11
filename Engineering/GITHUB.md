# DoG Five-Stage Dual-Stream

Industrial defect **closed-set multi-label classification + open-set rejection**: learnable **DoG contour + high-pass texture** → five-stage asymmetric dual-stream backbone.

## Layout

```
Engineering/
├── train.py                 # closed-set GC10 / NEU
├── open.py                  # open-set DoG (leave-one-out, experiments A/B/C/D)
├── exp.py                   # baselines, ablation, MSP, C+D dual-unknown
├── exp/                     # paper experiment scripts
├── utils/                   # model, data, training, evaluation
├── GC10-DET/  NEU-DET/      # datasets (not included in repo)
└── checkpoints/  results/   # outputs (.gitignore)
```

**Call chain:** `train.py` / `open.py` / `exp.py` → `exp/*` → `utils/*`

## Paper experiments (five groups)

| Paper block | Command |
|-------------|---------|
| **① Closed-set GC10 / NEU** | `python train.py --dataset gc10/neu --data-root . --deterministic` |
| **② Open-set + A/B/C/D** | `python open.py --experiment A --data-root . --deterministic` |
| **② C+D dual-unknown** | `python exp.py cd --data-root . --deterministic` |
| **③ Closed-set CNN baselines** | `python exp.py baseline --data-root . --deterministic` |
| **④ Ablation (Exp. A)** | `python exp.py ablation --data-root . --deterministic` |
| **⑤ Open-set MSP baseline** | `python exp.py msp --experiment A --data-root .` |

## Setup

```bash
cd Engineering
pip install -r requirements.txt
```

## Training (closed-set)

```bash
python train.py --dataset gc10 --data-root . --deterministic
python train.py --dataset gc10 --data-root . --no-tta   # disable TTA
python train.py --dataset neu --data-root . --deterministic
```

**Paper protocol:** `seed=44` + **`--deterministic`** for the main tables in the paper.

After training: early-stop epoch → checkpoint saved → metric breakdown (train selection / plain / paper) → closed-set summary.

**Data layout:**

```
GC10-DET/
  lable/          # class folders (note: folder name spelling)
  image/          # images
NEU-DET/
  images/  labels/
```

Closed-set checkpoint: `checkpoints/ification_opt_8_16_32_blurpool_dog_v2_best.pth`

**Split:** train / val / test = **60% / 20% / 20%** (`Config.val_size=0.2`, `test_size=0.2`, seed=44, stratified search over 256 candidates). Same protocol for closed-set, OSR, and baseline tables.

## Architecture (Sec. III)

```
Grayscale 1ch
  → LearnableDoGContourPrep (multi-scale DoG | Gaussian high-pass texture)
  → DualStreamNet, 5 stages (contour 5×5 / texture 3×3)
  → main + contour/texture auxiliary heads
```

**Loss:** ASL on main + aux heads (`L_aux`) + branch decoupling (`L_dec`) + DoG scale balancing. AdamW lr=5e-4, 100 epochs, seed=44.

## Open-set (OSR)

Train from scratch under semantic leave-one-out-unknown. Default **L_aux + L_dec**; prototype loss off (`--proto` optional). Rejection scores: main/aux entropy, JS divergence, `max_prob` (MSP-style).

```bash
python open.py --unknown-all --data-root . --deterministic
python open.py --experiment A --data-root . --deterministic   # B / C / D
python exp.py ablation --data-root . --deterministic
python exp.py cd --data-root . --deterministic                # C+D dual-unknown
```

### A/B/C/D selection (after full leave-one-out table)

| Exp | Unknown ID | Class | Criterion | Primary AUROC |
|-----|------------|-------|-----------|---------------|
| **A** | 10 | Inclusion | Highest main-head entropy (10 classes) | **0.9494** |
| **B** | 1 | Punching | Highest JS (10 classes) | **0.9298** |
| **C** | 6 | Pitted surface | 2nd-lowest main-head entropy | 0.4360 |
| **D** | 4 | Patch | Lowest main-head entropy | 0.3879 |

Closed-set Test μF1 (same `--deterministic` round): A 0.8037 · B 0.8210 · C 0.8358 · D 0.8654.

Checkpoints: `checkpoints/osr_dog_v2_unknown{K}_best.pth` · CSV: `results/osr_dog_v2_all_train.csv`

## ResNet-18 + MSP open-set baseline

Same leave-one-out protocol as `open.py` (60/20/20, seed=44, val T/τ). ResNet-18 scratch, **single ASL head**, rejection **MSP = 1 − max_k σ(z_k)** (same definition as `max_prob` on the DoG network).

| Exp | Unknown | Ours (primary) | ResNet+MSP |
|-----|---------|----------------|------------|
| A | 10 Inclusion | main entropy **0.9494** | 0.7848 |
| B | 1 Punching | JS **0.9298** | 0.5627 |
| C | 6 Pitted surface | main entropy 0.4360 | 0.6588 |
| D | 4 Patch | main entropy 0.3879 | 0.8051 |

```bash
python exp.py msp --experiment A --data-root .
```

Weights: `checkpoints/baseline_osr_resnet18_unknown{K}_best.pth`

## Closed-set CNN baselines

Fair comparison: **single-channel grayscale `[B,1,H,W]`**, stem `Conv2d(1→64, …)`, no RGB replication.

```bash
python exp.py baseline --data-root . --deterministic
python exp.py baseline --data-root . --pretrained --models resnet18
```

| Model | Params | Test μF1 |
|-------|--------|----------|
| **DoG dual-stream (ours)** | 44,824 | **0.8227** |
| ResNet-18 scratch | … | 0.758 |
| MobileNetV3-S/L scratch | … | 0.770 / 0.727 |

## Reproducibility

| Setting | Paper default | Notes |
|---------|---------------|-------|
| `seed` | **44** | split, init, shuffle |
| `--deterministic` | **on** for main tables | cudnn deterministic, workers=0 |
| `workers` | 0 (det) | default 2 without `--deterministic` |

## Dependencies

PyTorch, torchvision, scikit-learn, numpy, Pillow, etc. (see `requirements.txt`).
