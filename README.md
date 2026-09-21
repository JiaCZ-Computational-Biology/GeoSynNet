# 🧬 GeoSynNet

GeoSynNet is a geometry-aware multimodal deep learning framework that integrates 3D molecular geometry, topological fingerprints, and physicochemical descriptors for α-syn-associated activity prediction and virtual screening.

---

## 📁 Directory Structure

| Folder/File | Description |
|--------------|-------------|
| **`data/`** | Contains all datasets used in the experiments (full dataset, training set, validation set, and independent test set). |
| **`train.py`** | Final model training script. |
| **`test.py`** | Independent testing script. |
| **`ablation/`** | Includes all ablation experiment code and related materials. |
| **`gnn/`** | Contains implementations of the investigated graph neural networks. |
| **`fingerprint/`** | Includes all molecular fingerprints used for experimental comparisons. |

---

## ⚙️ Python Environment

| Library | Version |
|----------|----------|
| `python` | 3.8.18 |
| `numpy` | 1.24.4 |
| `pandas` | 2.0.3 |
| `pillow` | 10.4.0 |
| `torch` | 2.4.1 |
| `catboost` | 1.2.8 |
| `function` | 1.2.0 |
| `pycaret` | 3.2.0 |
| `xgboost` | 2.1.4 |

---
