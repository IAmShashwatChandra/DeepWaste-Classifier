# DeepWaste: Deep Learning Waste Classification & Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision pipeline designed for automated waste sorting. DeepWaste integrates deep feature extraction with bounding box localization and visual saliency maps (Grad-CAM) to provide transparent, interpretable classifications.

---

## 📌 Key Highlights & Performance

- **Backbone Architecture:** Fine-tuned **EfficientNet-B2** paired with a custom two-stage projection head.
- **Robust Augmentation:** Leveraged **Albumentations** (color jitter, shifts, rotations, scaling) to improve real-world generalization.
- **Localization & Interpretability:** Integrated **YOLOv8** for waste item detection alongside **Grad-CAM heatmaps** to verify spatial decision boundaries.
- **Benchmark Results:**

| Model Architecture | Input Resolution | Test Accuracy | Status |
| :--- | :---: | :---: | :---: |
| Baseline (ResNet-50) | 224x224 | 71.61% | Replaced |
| **DeepWaste (EfficientNet-B2)** | **224x224** | **80.82%** | **Deployed** |

---

## 🏗 Pipeline Architecture

```text
Input Image / Stream
       │
       ├──► [YOLOv8 Engine] ──► Multi-Object Bounding Boxes
       │
       └──► [Albumentations Preprocessing]
                 │
                 ▼
         [EfficientNet-B2 Backbone]
                 │
                 ▼
         [Two-Stage Projection Head] ──► Classification Prediction
                 │
                 └──► [Grad-CAM Layer] ──► Visual Heatmap Explanation
