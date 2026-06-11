# contour-texture-osr
Official PyTorch implementation of "A Contour-Texture Dual-Stream Measurement Framework with Learnable DoG for Open-Set Industrial Defect Inspection"
# Contour-Texture Dual-Stream Measurement Framework


Official PyTorch implementation of the paper:


**A Contour-Texture Dual-Stream Measurement Framework with Learnable DoG for Open-Set Industrial Defect Inspection**  


## Abstract


Industrial defect detection is fundamentally a visual measurement problem: it requires not only recognizing known defects but also producing interpretable anomaly scores for unknown defects. This work proposes a contour-texture dual-stream measurement framework that unifies closed-set classification and open-set rejection via learnable Difference of Gaussians (DoG) and path-adaptive anomaly scoring, achieving physically interpretable open-set defect inspection with only 45K parameters.


---


## ✨ Main Contributions


1. **Contour-Texture Physical Measurement Frontend**: Learnable DoG + fixed high-pass filtering decomposes images into contour and texture measurement maps, replacing ImageNet pre-training with physical priors


2. **Multi-Path Measurement Structure**: Main head + dual auxiliary heads enable open-set rejection without an extra rejection network, achieving 0.949/0.930 AUROC for Inclusion/Punching unknowns


3. **Path-Adaptive Rejection Principles**: Two effective rejection paths (fusion alertness and branch opposition) are revealed, with explicit measurement boundaries for semantically continuous unknowns


4. **Complete Experimental Validation**: GC10-DET open-set protocol, NEU-DET cross-dataset validation, and full reproducibility with deterministic training


---


## 📦 Requirements


```bash
pip install torch torchvision numpy pillow scikit-learn
