import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Display order (if the column exists in CSV, it will be shown)
DISPLAY_ORDER = ["accuracy", "precision", "recall", "auc", "dice", "iou"]

DISPLAY_NAME = {
    "accuracy": "ACCURACY (%)",
    "precision": "PRECISION (%)",
    "recall": "RECALL (%)",
    "auc": "AUC (%)",
    "dice": "DICE(%)",
    "iou": "IOU(%)",
}


def underline_cell(ax, fig, cell, pad_frac=0.08, y_frac=0.18, lw=1.2):
    """
    Draw underline inside a matplotlib table cell (matplotlib-version-safe).
    """
    renderer = fig.canvas.get_renderer()
    bbox = cell.get_window_extent(renderer=renderer)

    inv = ax.transAxes.inverted()
    (x0, y0) = inv.transform((bbox.x0, bbox.y0))
    (x1, y1) = inv.transform((bbox.x1, bbox.y1))

    pad = (x1 - x0) * pad_frac
    y = y0 + (y1 - y0) * y_frac

    ax.plot(
        [x0 + pad, x1 - pad],
        [y, y],
        transform=ax.transAxes,
        color="black",
        linewidth=lw,
        solid_capstyle="butt",
        zorder=10,
    )


def fmt_mean(mean, decimals=2, scale=100.0):
    """
    Format as mean only (no ±std), after scaling (default x100).
    """
    if mean is None:
        return "-"
    try:
        mean = float(mean)
        if np.isnan(mean):
            return "-"
    except Exception:
        return "-"
    mean *= scale
    return f"{mean:.{decimals}f}"


def top2_indices_desc(values: np.ndarray):
    """Return (best_idx, second_idx) for descending values, ignoring NaN."""
    values = values.astype(float)
    valid = np.where(~np.isnan(values))[0]
    if len(valid) == 0:
        return None, None
    order = valid[np.argsort(values[valid])[::-1]]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else None
    return best, second


def render_table_png(df: pd.DataFrame, metrics, out_png: str, decimals=2, title=None, scale=100.0):
    # Compute best/second using MEAN values
    best_idx = {}
    second_idx = {}
    for m in metrics:
        means = df[f"{m}_mean"].astype(float).to_numpy()
        b, s = top2_indices_desc(means)
        best_idx[m] = b
        second_idx[m] = s

    # Build table text (mean only)
    col_labels = ["Model"] + [DISPLAY_NAME.get(m, m.upper()) for m in metrics]
    cell_text = []
    for i in range(len(df)):
        row = [str(df.loc[i, "model"])]
        for m in metrics:
            row.append(fmt_mean(df.loc[i, f"{m}_mean"], decimals, scale=scale))
        cell_text.append(row)

    cols = len(col_labels)
    rows = len(df)
    fig_w = max(10, 1.6 * cols)
    fig_h = max(3.0, 0.55 * (rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)

    # Header bold
    for c in range(len(col_labels)):
        cell = table[(0, c)]
        cell.get_text().set_fontweight("bold")

    # Draw once so cell extents are available for underline
    fig.canvas.draw()

    # Apply styling: best=bold, second=underline
    for j, m in enumerate(metrics, start=1):
        b = best_idx[m]
        s = second_idx[m]
        if b is not None:
            cell = table[(b + 1, j)]
            cell.get_text().set_fontweight("bold")
        if s is not None:
            cell = table[(s + 1, j)]
            underline_cell(ax, fig, cell)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


def render_table_md(df: pd.DataFrame, metrics, out_md: str, decimals=2, scale=100.0):
    best_idx = {}
    second_idx = {}
    for m in metrics:
        means = df[f"{m}_mean"].astype(float).to_numpy()
        b, s = top2_indices_desc(means)
        best_idx[m] = b
        second_idx[m] = s

    headers = ["Model"] + [DISPLAY_NAME.get(m, m.upper()) for m in metrics]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for i in range(len(df)):
        row = [str(df.loc[i, "model"])]
        for m in metrics:
            val = fmt_mean(df.loc[i, f"{m}_mean"], decimals, scale=scale)
            if val != "-":
                if best_idx[m] == i:
                    val = f"**{val}**"
                elif second_idx[m] == i:
                    val = f"<ins>{val}</ins>"
            row.append(val)
        lines.append("| " + " | ".join(row) + " |")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Saved: {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="runs/metrics_table.csv")
    ap.add_argument("--out_png", type=str, default="runs/metrics_table.png")
    ap.add_argument("--out_md", type=str, default="runs/metrics_table.md")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--title", type=str, default="Model performance comparison on FaultSeg3D (Test)")
    ap.add_argument("--scale", type=float, default=100.0, help="Multiply all metrics by this factor (default 100)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Determine which metrics exist in CSV (still based on *_mean columns)
    metrics = []
    for m in DISPLAY_ORDER:
        if f"{m}_mean" in df.columns:
            metrics.append(m)

    if not metrics:
        raise RuntimeError(f"No metric columns like '*_mean' found in {args.csv}")

    render_table_png(df, metrics, args.out_png, decimals=args.decimals, title=args.title, scale=args.scale)
    render_table_md(df, metrics, args.out_md, decimals=args.decimals, scale=args.scale)


if __name__ == "__main__":
    main()
