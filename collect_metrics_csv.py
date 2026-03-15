# collect_metrics_csv.py
# Collect test_metrics.json from each run folder (defined by yaml output.dir)
# and write a single runs/metrics_table.csv that is compatible with render_metrics_table.py.

import os
import json
import argparse
from typing import List, Dict, Any

import yaml
import pandas as pd


DEFAULT_METRICS = ["accuracy", "precision", "recall", "auc", "dice", "iou"]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_metrics_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "test" not in obj:
        raise KeyError(f"'test' key not found in {path}")
    return obj["test"]


def metric_to_mean_std(v):
    """
    test_metrics.json format in your project is typically:
      "accuracy": [mean, std]
    This helper tolerates a few variants but always returns (mean, std) floats.
    """
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    # If someone saved only mean, keep std as NaN
    try:
        return float(v), float("nan")
    except Exception:
        return float("nan"), float("nan")


def build_rows(configs: List[str], metrics: List[str], strict: bool) -> List[Dict[str, Any]]:
    rows = []
    for cfg_path in configs:
        if not os.path.exists(cfg_path):
            msg = f"[WARN] config not found: {cfg_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        cfg = load_yaml(cfg_path)

        model_name = cfg.get("model", {}).get("name", os.path.splitext(os.path.basename(cfg_path))[0])
        out_dir = cfg.get("output", {}).get("dir", None)
        if out_dir is None:
            msg = f"[WARN] output.dir missing in: {cfg_path}"
            if strict:
                raise KeyError(msg)
            print(msg)
            continue

        metrics_path = os.path.join(out_dir, "test_metrics.json")
        if not os.path.exists(metrics_path):
            msg = f"[WARN] missing test_metrics.json: {metrics_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        test = load_test_metrics_json(metrics_path)

        row = {"model": str(model_name)}
        for k in metrics:
            if k not in test:
                msg = f"[WARN] metric '{k}' missing in {metrics_path}"
                if strict:
                    raise KeyError(msg)
                print(msg)
                mean, std = float("nan"), float("nan")
            else:
                mean, std = metric_to_mean_std(test[k])

            row[f"{k}_mean"] = mean
            row[f"{k}_std"] = std

        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/unet.yaml",
            "configs/resnet50_unet.yaml",
            "configs/transunet.yaml",
            "configs/transattunet.yaml",
            "configs/swinunet.yaml",
            "configs/transdeeplab.yaml",
            "configs/maxvitunet.yaml",
            "configs/deit3.yaml",
            "configs/convnextunet.yaml",
            "configs/segformer.yaml",
        ],
        help="List of config yaml paths to collect.",
    )
    ap.add_argument("--out_csv", type=str, default="runs/metrics_table.csv")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, missing config / missing json / missing metric will raise error instead of skipping.",
    )
    ap.add_argument(
        "--sort_by",
        type=str,
        default="",
        help="Optional: sort by a metric mean column, e.g. dice_mean or auc_mean. Empty keeps input order.",
    )
    args = ap.parse_args()

    rows = build_rows(args.configs, args.metrics, strict=args.strict)
    if not rows:
        raise RuntimeError("No rows collected. Check configs paths and runs/*/test_metrics.json existence.")

    df = pd.DataFrame(rows)

    # Optional sorting
    if args.sort_by:
        if args.sort_by not in df.columns:
            raise KeyError(f"--sort_by={args.sort_by} not found in columns: {list(df.columns)}")
        df = df.sort_values(by=args.sort_by, ascending=False, na_position="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df)
    print(f"[OK] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
