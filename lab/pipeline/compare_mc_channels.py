#!/usr/bin/env python3
"""Compare independent ChanA vs ChanB rigid shifts for one MC condition.

Usage
-----
    python lab/pipeline/compare_mc_channels.py <run_dir> \\
        --avg-a <stk_avg_a.tif> --avg-b <stk_avg_b.tif> \\
        --out <figure.png>

<run_dir> contains ChanA/ and ChanB/ with offsets.npz and ops.npy (meanImg).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread


def _load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        img = np.load(path)
    else:
        img = imread(str(path), is_ome=False)
    img = np.squeeze(np.asarray(img))
    if img.ndim == 3:
        img = img.mean(axis=0)
    return img


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def vertical_ridge_energy(img, ky_cut=0.05):
    """Fraction of 2D-FFT power at |ky| > ky_cut (PMT ridges live in Fourier-y)."""
    img = np.squeeze(np.asarray(img, dtype=np.float32))
    if img.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {img.shape}")
    img = img - img.mean()
    spec = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(spec) ** 2
    ly, _lx = power.shape
    cy, cx = ly // 2, power.shape[1] // 2
    power[cy, cx] = 0
    ky = (np.arange(ly) - cy) / float(ly)
    ridge = power[np.abs(ky) > ky_cut, :].sum()
    tot = power.sum()
    return float(ridge / (tot + 1e-12))


def load_channel(chan_dir):
    chan_dir = Path(chan_dir)
    z = np.load(chan_dir / "offsets.npz")
    ops = np.load(chan_dir / "ops.npy", allow_pickle=True).item()
    return {
        "yoff": z["yoff"],
        "xoff": z["xoff"],
        "cmax": z["cmax"],
        "mean": ops["meanImg"],
    }


def compare(run_dir, avg_a=None, avg_b=None, out_path=None):
    run_dir = Path(run_dir)
    a = load_channel(run_dir / "ChanA")
    b = load_channel(run_dir / "ChanB")
    stats = {
        "pearson_xoff": _pearson(a["xoff"], b["xoff"]),
        "pearson_yoff": _pearson(a["yoff"], b["yoff"]),
        "cmax_median_A": float(np.median(a["cmax"])),
        "cmax_median_B": float(np.median(b["cmax"])),
        "ridge_reg_A": vertical_ridge_energy(a["mean"]),
        "ridge_reg_B": vertical_ridge_energy(b["mean"]),
    }
    if avg_a is None:
        avg_a = run_dir / "ChanA" / "mean_unregistered.npy"
    if avg_b is None:
        avg_b = run_dir / "ChanB" / "mean_unregistered.npy"
    if avg_a is not None and Path(avg_a).exists():
        avg_a_img = _load_image(avg_a)
        stats["ridge_avg_A"] = vertical_ridge_energy(avg_a_img)
    else:
        avg_a_img = None
    if avg_b is not None and Path(avg_b).exists():
        avg_b_img = _load_image(avg_b)
        stats["ridge_avg_B"] = vertical_ridge_energy(avg_b_img)
    else:
        avg_b_img = None

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    t = np.arange(len(a["yoff"]))
    axes[0, 0].plot(t, a["xoff"], label="A", lw=0.8)
    axes[0, 0].plot(t, b["xoff"], label="B", lw=0.8, alpha=0.85)
    axes[0, 0].set_title(f"xoff   r={stats['pearson_xoff']:.3f}")
    axes[0, 0].legend()
    axes[0, 1].plot(t, a["yoff"], label="A", lw=0.8)
    axes[0, 1].plot(t, b["yoff"], label="B", lw=0.8, alpha=0.85)
    axes[0, 1].set_title(f"yoff   r={stats['pearson_yoff']:.3f}")
    axes[0, 2].scatter(a["xoff"], b["xoff"], s=2, alpha=0.3)
    axes[0, 2].set_title("xoff A vs B")
    axes[0, 2].set_xlabel("A")
    axes[0, 2].set_ylabel("B")
    axes[0, 3].scatter(a["yoff"], b["yoff"], s=2, alpha=0.3)
    axes[0, 3].set_title("yoff A vs B")
    axes[0, 3].set_xlabel("A")
    axes[0, 3].set_ylabel("B")

    axes[1, 0].plot(t, a["cmax"], lw=0.6)
    axes[1, 0].set_title(f"cmax A  med={stats['cmax_median_A']:.4f}")
    axes[1, 1].plot(t, b["cmax"], lw=0.6, color="C1")
    axes[1, 1].set_title(f"cmax B  med={stats['cmax_median_B']:.4f}")
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    lines = [
        f"xoff r = {stats['pearson_xoff']:.3f}",
        f"yoff r = {stats['pearson_yoff']:.3f}   (fringe-lock ~0.07)",
        f"cmax A/B med = {stats['cmax_median_A']:.4f} / {stats['cmax_median_B']:.4f}",
        f"ridge energy (Fourier-y) registered A/B = "
        f"{stats['ridge_reg_A']:.4f} / {stats['ridge_reg_B']:.4f}",
    ]
    if "ridge_avg_A" in stats:
        lines.append(
            f"ridge energy stk_avg A/B = {stats['ridge_avg_A']:.4f} / {stats.get('ridge_avg_B', float('nan')):.4f}"
        )
        lines.append("Registered ridge should not exceed stk_avg.")
    axes[1, 2].text(0.0, 0.5, "\n".join(lines), va="center", family="monospace", fontsize=9)
    axes[1, 2].set_xlim(0, 1)

    def _show(ax, img, title):
        if img is None:
            ax.axis("off")
            return
        lo, hi = np.percentile(img, (1, 99))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")

    _show(axes[2, 0], avg_a_img, "stk_avg A")
    _show(axes[2, 1], a["mean"], "registered mean A")
    _show(axes[2, 2], avg_b_img, "stk_avg B")
    _show(axes[2, 3], b["mean"], "registered mean B")
    fig.suptitle(str(run_dir), fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path or (run_dir / "compare_AB.png"))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    stats_path = out_path.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_path}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--avg-a", default=None)
    p.add_argument("--avg-b", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    compare(args.run_dir, avg_a=args.avg_a, avg_b=args.avg_b, out_path=args.out)


if __name__ == "__main__":
    main()
