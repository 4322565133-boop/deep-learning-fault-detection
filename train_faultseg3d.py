import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from options import parse_args
from utils.seed import set_seed
from utils.io import ensure_dir, SimpleLogger, save_json
from dataset.FaultSeg3D_Dataset import FaultSeg3D2DSlices
from models.factory import build_model
from losses.bce_dice import BCEDiceLoss
from model_evaluate.metrics import (
    confusion_from_binary, accuracy, precision, recall, roc_auc_score_binary, dice_coeff, iou_coeff
)

# Enable anomaly detection to find bad gradients (optional, can be turned off for speed)
torch.autograd.set_detect_anomaly(False)


def get_device(pref: str = ""):
    """Select the computing device (CUDA, MPS, or CPU)."""
    pref = (pref or "").lower().strip()
    if pref in ["cuda", "cpu", "mps"]:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def split_val_test_ids(val_split_dir: str, val_count: int, test_count: int):
    """
    Split the volumes in the validation folder into validation set and test set based on filenames.
    """
    seis_dir = os.path.join(val_split_dir, "seis")
    # Get all volume IDs (filenames without extension)
    ids = sorted([os.path.splitext(f)[0] for f in os.listdir(seis_dir) if f.endswith(".dat") and (not f.startswith("."))])
    assert len(ids) >= val_count + test_count, f"Validation folder has {len(ids)} volumes, but need >= {val_count+test_count}"
    
    # Return two lists: one for validation, one for final testing
    return ids[:val_count], ids[val_count:val_count+test_count]

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Run evaluation on a dataset and return metrics (mean and std).
    """
    model.eval()
    accs, precs, recs, aucs, dices, ious = [], [], [], [], [], []
    
    for img, mask, meta in loader:
        img = img.to(device)
        mask = mask.to(device)
        
        # Forward pass
        logits = model(img)
        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()

        # Convert to numpy for metric calculation
        p = pred.detach().cpu().numpy()
        gt = mask.detach().cpu().numpy()
        pr = probs.detach().cpu().numpy()

        # Calculate metrics for each image in the batch
        B = p.shape[0]
        for b in range(B):
            tp, tn, fp, fn = confusion_from_binary(p[b], gt[b])
            accs.append(accuracy(tp, tn, fp, fn))
            precs.append(precision(tp, tn, fp, fn))
            recs.append(recall(tp, tn, fp, fn))
            dices.append(dice_coeff(p[b], gt[b]))
            ious.append(iou_coeff(p[b], gt[b]))
            aucs.append(roc_auc_score_binary(gt[b], pr[b]))

    def mean_std(arr):
        """Helper to calculate mean and std, handling NaNs."""
        arr = np.asarray(arr, dtype=np.float64)
        if np.isnan(arr).any():
            arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std(ddof=0))

    return {
        "accuracy": mean_std(accs),
        "precision": mean_std(precs),
        "recall": mean_std(recs),
        "auc": mean_std(aucs),
        "dice": mean_std(dices),
        "iou": mean_std(ious),
    }

def train_one_epoch(model, loader, optim, loss_fn, device, grad_clip=0.0):
    """
    Run one epoch of training.
    """
    model.train()
    losses = []
    
    for img, mask, meta in loader:
        img = img.to(device)
        mask = mask.to(device)
        
        optim.zero_grad(set_to_none=True)
        logits = model(img)
        loss = loss_fn(logits, mask)
        loss.backward()
        
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optim.step()
        losses.append(loss.item())
        
    return float(np.mean(losses)) if losses else float("nan")

def main():
    # 1. Setup Configuration and Environment
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set random seed for reproducibility
    set_seed(cfg["data"].get("seed", 42), deterministic=False)
    device = get_device(args.device)

    # Create output directory and logger
    out_dir = ensure_dir(cfg["output"]["dir"])
    logger = SimpleLogger(os.path.join(out_dir, "train.log"))
    logger.log(f"Device: {device}")
    logger.log(f"Config: {args.config}")

    # 2. Prepare Data
    data_root = cfg["data"]["root"]
    train_dir = os.path.join(data_root, cfg["data"]["train_split"])
    val_dir = os.path.join(data_root, cfg["data"]["val_split"])

    # Split the 'validation' folder into validation set (for best model selection) and test set (for final benchmark)
    val_ids, test_ids = split_val_test_ids(val_dir, cfg["data"]["val_count"], cfg["data"]["test_count"])
    # Save the split IDs to ensure we know exactly which volumes were used for what
    save_json({"val_ids": val_ids, "test_ids": test_ids}, os.path.join(out_dir, "val_test_split.json"))

    # --- Train Dataset: Uses 'train' mode (Random sampling per epoch) ---
    train_ds = FaultSeg3D2DSlices(
        split_dir=train_dir,
        vol_shape=tuple(cfg["data"]["vol_shape"]),
        axes=cfg["data"]["axes"],
        img_size=cfg["data"]["img_size"],
        is_train=True,
        train_repeats=cfg["data"]["train_repeats"],
        mode="train",  # Random slicing for training
        aug_p=cfg["train"]["aug_p"],
        seed=cfg["data"].get("seed", 42),
    )

    # --- Validation Dataset: Uses 'eval_fixed' mode (Fixed slices per epoch) ---
    # We generate a fixed list of slices to ensure validation metrics are comparable across epochs.
    print("Generating fixed sampling plan for validation...")
    val_samples = []
    # Use a specific random state to ensure the list is deterministic
    rng = np.random.RandomState(cfg["data"].get("seed", 42))
    vol_dims = cfg["data"]["vol_shape"]
    valid_axes = cfg["data"]["axes"]
    samples_per_vol = 50 # Number of fixed slices to check per volume

    for vid in val_ids:
        for _ in range(samples_per_vol):
            ax = rng.choice(valid_axes)
            # Ensure we select a valid slice index for the chosen axis
            max_slice = vol_dims[ax]
            sl = rng.randint(0, max_slice)
            val_samples.append((vid, int(ax), int(sl)))

    val_ds = FaultSeg3D2DSlices(
        split_dir=val_dir,
        ids=val_ids,
        vol_shape=tuple(cfg["data"]["vol_shape"]),
        axes=cfg["data"]["axes"],
        img_size=cfg["data"]["img_size"],
        is_train=False,
        train_repeats=1,
        mode="eval_fixed",          # Use fixed samples
        fixed_samples=val_samples,  # Pass the generated list
        aug_p=0.0,                  # No augmentation for validation
        seed=cfg["data"].get("seed", 42),
    )

    # --- Test Dataset: Uses 'eval_enum' mode (All possible slices) ---
    # For the final test, we want to evaluate on ALL slices to get the most accurate metric.
    test_ds = FaultSeg3D2DSlices(
        split_dir=val_dir,
        ids=test_ids,
        vol_shape=tuple(cfg["data"]["vol_shape"]),
        axes=cfg["data"]["axes"],
        img_size=cfg["data"]["img_size"],
        is_train=False,
        train_repeats=1,
        mode="eval_enum",  # Enumerate all slices
        aug_p=0.0,
        seed=cfg["data"].get("seed", 42),
    )

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"], drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"]["num_workers"], drop_last=False)

    # 3. Build Model
    model_cfg = cfg["model"]
    model = build_model(
        model_cfg["name"],
        n_channels=model_cfg.get("n_channels", 1),
        n_classes=model_cfg.get("n_classes", 1),
        pretrained=bool(model_cfg.get("pretrained", False)),
        **(model_cfg.get("kwargs", {}) or {})
    ).to(device)

    # 4. Evaluation Only Mode (if requested)
    if args.eval_only:
        ckpt_path = args.ckpt or os.path.join(out_dir, "best.pt")
        logger.log(f"Eval-only mode. Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        metrics = evaluate(model, test_loader, device, threshold=0.5)
        save_json({"test": metrics}, os.path.join(out_dir, "test_metrics.json"))
        logger.log(f"Test metrics: {metrics}")
        return

    # 5. Optimizer and Loss Setup
    loss_fn = BCEDiceLoss(1.0, 1.0)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    # 6. Training Loop
    best_score = -1.0
    history = []

    logger.log("Starting training loop...")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optim, loss_fn, device, grad_clip=cfg["train"].get("grad_clip", 0.0))
        
        # Validate (on fixed set)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        val_dice_mean = val_metrics["dice"][0]
        
        # Save best model
        is_best = val_dice_mean > best_score
        if is_best:
            best_score = val_dice_mean
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_score": best_score}, os.path.join(out_dir, "best.pt"))
        
        # Save last model
        torch.save({"model": model.state_dict(), "epoch": epoch, "best_score": best_score}, os.path.join(out_dir, "last.pt"))

        # Log history
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc_mean": val_metrics["accuracy"][0],
            "val_prec_mean": val_metrics["precision"][0],
            "val_rec_mean": val_metrics["recall"][0],
            "val_auc_mean": val_metrics["auc"][0],
            "val_dice_mean": val_metrics["dice"][0],
            "val_iou_mean": val_metrics["iou"][0],
            "seconds": dt,
            "best": int(is_best),
        }
        history.append(row)
        logger.log(f"Epoch {epoch:03d} loss={train_loss:.4f} val_dice={val_dice_mean:.4f} best={best_score:.4f} time={dt:.1f}s")

    # Save history to CSV
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # 7. Final Test
    # Load the best model found during training
    logger.log("Loading best model for final testing...")
    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    
    # Run evaluation on Test Set (using enumeration mode for full coverage)
    test_metrics = evaluate(model, test_loader, device, threshold=0.5)
    save_json({"test": test_metrics}, os.path.join(out_dir, "test_metrics.json"))
    logger.log(f"Final Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()