# JINTAN/infer_jintan_models.py
import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # Added for solid color mapping
import segyio

from models.factory import build_model


MODEL_RUNS = [
    # (display_name, config_path, checkpoint_path)
    ("UNet", "configs/unet.yaml", "runs/unet/best.pt"),
    ("ResNet50_UNet", "configs/resnet50_unet.yaml", "runs/resnet50_unet/best.pt"),
    ("TransUNet", "configs/transunet.yaml", "runs/transunet/best.pt"),
    ("TransAttUNet", "configs/transattunet.yaml", "runs/transattunet/best.pt"),
    ("SwinUNet", "configs/swinunet.yaml", "runs/swinunet/best.pt"),
    ("TransDeepLab", "configs/transdeeplab.yaml", "runs/transdeeplab/best.pt"),
    ("MaxViTUNet", "configs/maxvitunet.yaml", "runs/maxvitunet/best.pt"),
    ("ConvNeXtUNet", "configs/convnextunet.yaml", "runs/convnextunet/best.pt"),
    ("SegFormer", "configs/segformer.yaml", "runs/segformer/best.pt"),
]


def get_device(pref: str = ""):
    pref = (pref or "").lower().strip()
    if pref in ["cuda", "cpu", "mps"]:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def robust_norm_to_minus1_1(sec: np.ndarray, p_low=2.0, p_high=98.0):
    """sec: (H,W) float32 -> normalized to [-1,1] and also returns display [0,1]."""
    lo = np.percentile(sec, p_low)
    hi = np.percentile(sec, p_high)
    if hi <= lo:
        lo, hi = float(sec.min()), float(sec.max() + 1e-6)
    x = np.clip(sec, lo, hi)
    x01 = (x - lo) / (hi - lo + 1e-6)  # [0,1]
    x11 = x01 * 2.0 - 1.0              # [-1,1]
    return x11.astype(np.float32), x01.astype(np.float32)


def read_traceblock(sgy_path: str, start_trace: int, n_traces: int):
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        f.mmap()
        n_total = f.tracecount
        ns = len(f.samples)

        start = int(start_trace)
        start = max(0, min(start, n_total - 1))
        end = min(start + int(n_traces), n_total)

        traces = np.stack([f.trace[i] for i in range(start, end)], axis=0)  # (W, H)
        sec = traces.T  # (H, W)

        info = {"tracecount": n_total, "nsamples": ns, "start": start, "end": end, "n_traces": end - start}
        return sec.astype(np.float32), info


def safe_load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def infer_stitch(model, x11: np.ndarray, patch=224, stride=112, batch_size=16, device=torch.device("cpu")):
    """
    x11: (H,W) in [-1,1]
    returns prob: (H,W) in [0,1]
    """
    model.eval()

    H, W = x11.shape
    patch = int(patch)
    stride = int(stride)

    # pad to cover borders
    pad_h = (patch - H % stride) % stride
    pad_w = (patch - W % stride) % stride
    pad_top = patch // 2
    pad_left = patch // 2
    pad_bottom = pad_top + pad_h
    pad_right = pad_left + pad_w

    x_pad = np.pad(x11, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    Hp, Wp = x_pad.shape

    acc = np.zeros((Hp, Wp), dtype=np.float32)
    cnt = np.zeros((Hp, Wp), dtype=np.float32)

    # generate patch coords
    ys = list(range(0, Hp - patch + 1, stride))
    xs = list(range(0, Wp - patch + 1, stride))

    coords = [(y, x) for y in ys for x in xs]

    def run_batch(batch_coords):
        patches = []
        for (y, x) in batch_coords:
            p = x_pad[y:y+patch, x:x+patch]  # (patch,patch)
            patches.append(p)
        inp = torch.from_numpy(np.stack(patches, 0)).unsqueeze(1).to(device)  # (B,1,patch,patch)
        logits = model(inp)
        prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (B,patch,patch)
        for (y, x), pr in zip(batch_coords, prob):
            acc[y:y+patch, x:x+patch] += pr
            cnt[y:y+patch, x:x+patch] += 1.0

    for i in range(0, len(coords), batch_size):
        run_batch(coords[i:i+batch_size])

    prob_pad = acc / np.maximum(cnt, 1e-6)

    # crop back to original
    prob = prob_pad[pad_top:pad_top+H, pad_left:pad_left+W]
    return prob


def save_overlay(seis01: np.ndarray, prob: np.ndarray, out_png: str, title: str, thr=0.5):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    plt.figure(figsize=(12, 4))
    
    # 1. Plot the background seismic image
    plt.imshow(seis01, cmap="gray", aspect="auto")
    
    # 2. Convert probability map to a strict binary mask (0 or 1)
    binary_pred = (prob >= thr).astype(float)
    
    # 3. Mask the background (0s) so only the fault lines (1s) are plotted
    masked_pred = np.ma.masked_where(binary_pred == 0, binary_pred)
    
    # 4. Create a custom colormap that contains exactly ONE color (solid red)
    solid_red_cmap = ListedColormap(['red'])
    
    # 5. Plot the binary mask. 
    # - alpha=0.65 reduces the opacity for a lighter color that still 
    #   stands out but lets more background through, making it feel less 'deep'.
    # - Removing interpolation="none" allows matplotlib to anti-alias the edges
    #   slightly, further reducing the harshness/depth of the lines.
    plt.imshow(
        masked_pred, 
        cmap=solid_red_cmap, 
        alpha=0.65, 
        aspect="auto"
    )
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    # 6. Increase DPI slightly for crisper rendering in papers
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgy", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="JINTAN/outputs/fault_overlay")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--block", type=int, default=800)
    ap.add_argument("--patch", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--p_low", type=float, default=2.0)
    ap.add_argument("--p_high", type=float, default=98.0)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[INFO] device={device}")

    sec, info = read_traceblock(args.sgy, args.start, args.block)
    x11, seis01 = robust_norm_to_minus1_1(sec, p_low=args.p_low, p_high=args.p_high)

    tag = f"traceblock_{info['start']}_{info['end']-1}"
    print(f"[INFO] section={tag}, shape={sec.shape}")

    for disp, cfg_path, ckpt_path in MODEL_RUNS:
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {disp}: checkpoint not found: {ckpt_path}")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg["model"]

        model = build_model(
            mcfg["name"],
            n_channels=mcfg.get("n_channels", 1),
            n_classes=mcfg.get("n_classes", 1),
            pretrained=bool(mcfg.get("pretrained", False)),
            **(mcfg.get("kwargs", {}) or {}),
        ).to(device)

        model = safe_load_checkpoint(model, ckpt_path, device)

        prob = infer_stitch(
            model,
            x11,
            patch=args.patch,
            stride=args.stride,
            batch_size=args.batch,
            device=device,
        )

        out_png = os.path.join(args.out_dir, f"{disp}_{tag}_thr{args.thr:.2f}.png")
        title = f"{disp} on JINTAN ({tag})  patch={args.patch} stride={args.stride} thr={args.thr:.2f}"
        save_overlay(seis01, prob, out_png, title, thr=args.thr)
        print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()