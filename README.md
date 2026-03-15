# 🌍 FaultSeg3D & Beyond: Deep Learning for 3D Seismic Fault Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the official repository of my research on 3D seismic fault segmentation. 

While many papers focus purely on academic benchmarks, the true test of a geological deep learning model lies in its robustness against highly noisy, real-world seismic volumes. This project bridges that gap. Here, I built a highly modular pipeline from scratch to benchmark **9 state-of-the-art (SOTA) architectures** (CNNs, ViTs, and Custom Hybrids) on the public FaultSeg3D dataset, and successfully deployed the ultimate winning model (**MaxViTUNet**) onto an extremely challenging real-world dataset from the Sarawak Basin, Malaysia.

---

## 🚀 1. The Benchmark: CNNs vs. Vision Transformers

To find the optimal architecture for fault detection, I implemented a unified, highly-optimized 2.5D slicing pipeline (with LRU Caching for I/O bottleneck mitigation) to train and evaluate 9 models under strictly controlled, deterministic conditions.

**The Contenders:**
* **Classical CNNs:** U-Net, ResNet50_UNet, ConvNeXtUNet
* **Vision Transformers & Hybrids:** TransUNet, SwinUNet, SegFormer, MaxViTUNet, TransDeepLab (Swin + ASPP), and my custom **TransAttUNet** (Dual-Attention Bottleneck).

### 📊 Quantitative Results (Test Set)
All models were trained using a combined `BCE + Dice` loss. As shown below, **MaxViTUNet** emerged as the definitive champion, achieving the highest **DICE (69.88%)** and **IoU (55.72%)**. This proves that the hierarchical feature extraction of CNNs combined with the global attention mechanism of ViTs is the key to capturing long, continuous fault structures.

![Metrics Table](docs/figures/metrics_table.png)

### 🗺️ Qualitative Grid
A visual comparison of the predictions. The hybrid architectures (especially MaxViTUNet and TransAttUNet) produce significantly cleaner and more continuous fault lines compared to the fragmented predictions of pure CNNs.

![Qualitative Grid](docs/figures/qualitative_grid_test.png)

---

## 🏭 2. Real-World Deployment: JINTAN Oilfield (Malaysia)

Benchmarking is not the end. To prove the engineering value of this research, I deployed the trained MaxViTUNet onto a commercial `.sgy` seismic dataset (JINTAN) from the **Sarawak Basin, Malaysia**. 

This dataset is plagued with severe geological noise and chaotic reflections. I implemented a robust `infer_stitch` sliding-window inference algorithm to seamlessly reconstruct the high-resolution fault maps.

By applying a high confidence threshold (`thr=0.80`), the model successfully filters out the noise and crisply delineates the primary fault architectures.

| Raw Seismic Cross-Section | MaxViTUNet Prediction (Threshold = 0.80) |
| :---: | :---: |
| ![Preview](docs/figures/preview_traceblock_432693_433692.png) | <img src="docs/figures/MaxViTUNet_traceblock_432693_433692_thr0.80.png" width="100%"> |

*Notice how the model completely ignores the horizontal stratigraphic noise and precisely captures the nearly vertical fault lines in crisp, solid red.*

---

## 🛠️ Getting Started (Local Deployment)

If you want to reproduce my experiments or apply these models to your own geological data, follow these steps.

### Prerequisites
* **OS:** Linux or macOS (Apple Silicon MPS is fully supported)
* **Python:** 3.8+
* **GPU:** NVIDIA GPU with CUDA (Highly Recommended) or Mac M-series

### 1. Clone & Environment Setup
```bash
git clone [https://github.com/4322565133-boop/deep-learning-fault-detection.git](https://github.com/4322565133-boop/deep-learning-fault-detection.git)
cd deep-learning-fault-detection

# Initialize virtual environment
python -m venv .venv
source .venv/bin/activate

# Install strictly defined dependencies
pip install -U pip
pip install -r requirements.txt
```

### 2. Dataset Preparation
For the benchmark, download the `FaultSeg3D` dataset and place it in `./data/wangjing/`. The pipeline expects the following structure (raw `.dat` float32 volumes):
```text
data/wangjing/
├── train/
│   ├── seis/   # <id>.dat
│   └── fault/  # <id>.dat
└── validation/
    ├── seis/
    └── fault/
```

### 3. Training & Evaluation
Everything is configuration-driven. To train a specific model (e.g., my custom TransAttUNet):
```bash
python train_faultseg3d.py --config configs/transattunet.yaml
```

To run the automated pipeline that trains all 9 models and generates the CSV metrics table:
```bash
python benchmark_all.py
```

### 4. Run Inference on Your Own `.sgy` Data
Want to test the model on your own SEGY files? Use the dedicated JINTAN inference script. It handles raw SEGY amplitude normalization, slicing, and smooth patching automatically.

```bash
python JINTAN/infer_jintan_models.py \
    --sgy /path/to/your/seismic.sgy \
    --start 432693 \
    --block 1000 \
    --thr 0.80
```
*The output will be saved in `JINTAN/outputs/fault_overlay/` as a high-resolution PNG.*

---

## 🤝 Citation & Contact
If you find this codebase or the pre-trained weights useful for your research or industrial applications, please consider giving this repository a ⭐. Feel free to open an issue if you encounter any problems!