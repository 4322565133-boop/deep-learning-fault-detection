# JINTAN/preview_jintan_sgy.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import segyio


def robust_scale(x: np.ndarray, p_low=2.0, p_high=98.0):
    """Robust display scaling using percentiles."""
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    x = np.clip(x, lo, hi)
    return x, lo, hi


def read_traceblock(sgy_path: str, start_trace: int, n_traces: int):
    """
    Read a trace block as a 2D section: shape (nsamples, n_traces).
    Uses ignore_geometry=True to survive messy inline/xline headers.
    """
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        f.mmap()  # speed up random trace access
        n_total = f.tracecount
        ns = len(f.samples)

        start = int(start_trace)
        start = max(0, min(start, n_total - 1))
        end = min(start + int(n_traces), n_total)

        # segyio returns each trace as (nsamples,)
        traces = np.stack([f.trace[i] for i in range(start, end)], axis=0)  # (n_traces, nsamples)
        sec = traces.T  # (nsamples, n_traces)

        info = {
            "tracecount": n_total,
            "nsamples": ns,
            "start": start,
            "end": end,
            "n_traces": end - start,
        }
        return sec.astype(np.float32), info


def pick_starts(n_total: int, block: int, n: int, strategy: str, start: int, stride: int):
    """
    Choose start indices for preview blocks.
    - linspace: spread across whole file to avoid seeing similar adjacent blocks
    - stride: start + k*stride
    """
    block = int(block)
    n = int(n)
    max_start = max(0, n_total - block)

    if strategy == "linspace":
        if n <= 1:
            return [min(max_start, max(0, int(start)))]
        starts = np.linspace(0, max_start, n, dtype=int).tolist()
        return starts

    # default: stride
    base = 0 if start is None else int(start)
    if stride is None or int(stride) <= 0:
        stride = block  # avoid overlap by default
    starts = [min(max_start, max(0, base + k * int(stride))) for k in range(n)]
    return starts


def save_section_png(sec: np.ndarray, out_png: str, title: str, p_low=2.0, p_high=98.0):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    sec_disp, lo, hi = robust_scale(sec, p_low=p_low, p_high=p_high)

    plt.figure(figsize=(12, 4))
    plt.imshow(sec_disp, cmap="gray", aspect="auto", vmin=lo, vmax=hi)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgy", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="JINTAN/outputs/preview")
    ap.add_argument("--block", type=int, default=800, help="number of traces per section")
    ap.add_argument("--n", type=int, default=6, help="how many preview images")
    ap.add_argument("--strategy", type=str, default="linspace", choices=["linspace", "stride"])
    ap.add_argument("--start", type=int, default=0, help="used when strategy=stride")
    ap.add_argument("--stride", type=int, default=800, help="used when strategy=stride")
    ap.add_argument("--p_low", type=float, default=2.0)
    ap.add_argument("--p_high", type=float, default=98.0)
    args = ap.parse_args()

    # get tracecount quickly
    with segyio.open(args.sgy, "r", ignore_geometry=True) as f:
        n_total = f.tracecount
        ns = len(f.samples)
    print(f"[INFO] tracecount={n_total}, nsamples={ns}")

    starts = pick_starts(n_total, args.block, args.n, args.strategy, args.start, args.stride)

    for s in starts:
        sec, info = read_traceblock(args.sgy, s, args.block)
        tag = f"traceblock_{info['start']}_{info['end']-1}"
        out_png = os.path.join(args.out_dir, f"preview_{tag}.png")
        title = f"JINTAN preview: {tag}  shape={sec.shape}  (p{args.p_low}-p{args.p_high})"
        save_section_png(sec, out_png, title, p_low=args.p_low, p_high=args.p_high)
        print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()