# FaultSeg3D Benchmark (6 Models)

Train & compare 6 segmentation models on **FaultSeg3D** using **3D volumes -> on-the-fly 2D slices**.

Models:
- UNet
- ResNet50_UNet (ImageNet pretrained)
- TransUNet (ResNet50 + Transformer bottleneck, ImageNet pretrained)
- TransAttUNet (train from scratch)
- SwinUNet (ImageNet pretrained via timm)
- TransDeepLab (Swin backbone + ASPP, ImageNet pretrained via timm)

## Dataset layout

Put data under:

```
FAULTSEG3D/
  data/
    wangjing/
      train/
        seis/   <id>.dat
        fault/  <id>.dat
      validation/
        seis/   <id>.dat
        fault/  <id>.dat
```

- `train`: 200 volumes
- `validation`: 20 volumes -> **first 15 (sorted ids) used for val**, **next 5 used for test**

Each `.dat` is float32 volume (default `128x128x128`).

## Setup (macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Script auto-selects `cuda` -> `mps` -> `cpu`.

## Train one model

```bash
python train_faultseg3d.py --config configs/unet.yaml
```

Outputs go to `runs/<model>/`.

## Train all 6 models

```bash
python benchmark_all.py
```

Produces `runs/metrics_table.csv`.

## Training details
- Optimizer: AdamW (betas 0.9/0.999, weight_decay 0.02, lr 1e-3)
- Loss: BCE + Dice (1:1)
- Threshold: 0.5
- Epochs: 10 (smoke test; increase later)


## 5) Qualitative grid (paper-like Figure 6)

After training all models (so each has `runs/<model>/best.pt`):

```bash
python visualize_qualitative.py --data_root ./data/wangjing --out runs/qualitative_grid.png
```

## 6) Render metrics table image (paper-like Table 2)

After `benchmark_all.py` generates `runs/metrics_table.csv`:

```bash
python render_metrics_table.py --csv runs/metrics_table.csv --out runs/metrics_table.png
```
