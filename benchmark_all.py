import os, json, subprocess
import yaml
import pandas as pd

MODEL_CONFIGS = [
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
]


def main():
    # If you do NOT want to retrain all models, temporarily keep only the new config here
    # or comment out this training loop.
    for cfg in MODEL_CONFIGS:
        print(f"=== Running {cfg} ===")
        subprocess.check_call(["python", "train_faultseg3d.py", "--config", cfg])

    rows = []
    for cfg in MODEL_CONFIGS:
        with open(cfg, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        out_dir = c["output"]["dir"]
        with open(os.path.join(out_dir, "test_metrics.json"), "r", encoding="utf-8") as f:
            m = json.load(f)["test"]

        row = {"model": c["model"]["name"]}
        for k in ["accuracy", "precision", "recall", "auc", "dice", "iou"]:
            mean, std = m[k]
            row[f"{k}_mean"] = mean
            row[f"{k}_std"] = std

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("runs", exist_ok=True)
    df.to_csv("runs/metrics_table.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
