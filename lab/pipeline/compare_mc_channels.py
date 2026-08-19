#!/usr/bin/env python3
"""Compare independent ChanA vs ChanB rigid shifts for one MC condition.

Default image scores (2026-08-19) are signature-mask **fringe** vs mid-band
**cell** power (`lab.pipeline.mc_fft_metrics`). Legacy |ky|>0.05 ridge is
kept as a secondary field.

Usage
-----
    python lab/pipeline/compare_mc_channels.py <run_dir> \\
        --avg-a <unreg_a> --avg-b <unreg_b> \\
        --out <figure.png>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lab.pipeline.mc_fft_metrics import (
    find_default_signature,
    flatten_channel_scores,
    load_signature,
    score_pair,
    vertical_ridge_energy,
)


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


def _score_channel(letter, unreg, reg, run_dir, signature_path):
    if unreg is None:
        return {
            f"ridge_reg_{letter}": vertical_ridge_energy(reg),
            f"verdict_{letter}": "no_unreg",
            f"signature_{letter}": None,
        }
    if signature_path is None:
        signature_path = find_default_signature(letter, run_dir)
    if signature_path is None or not Path(signature_path).exists():
        return {
            f"ridge_reg_{letter}": vertical_ridge_energy(reg),
            f"ridge_unreg_{letter}": vertical_ridge_energy(unreg),
            f"verdict_{letter}": "no_signature",
            f"signature_{letter}": None,
        }
    pair = score_pair(unreg, reg, load_signature(signature_path), signature_path)
    return flatten_channel_scores(letter, pair)


def compare(
    run_dir,
    avg_a=None,
    avg_b=None,
    out_path=None,
    signature_a=None,
    signature_b=None,
):
    run_dir = Path(run_dir)
    a = load_channel(run_dir / "ChanA")
    b = load_channel(run_dir / "ChanB")
    stats = {
        "pearson_xoff": _pearson(a["xoff"], b["xoff"]),
        "pearson_yoff": _pearson(a["yoff"], b["yoff"]),
        "cmax_median_A": float(np.median(a["cmax"])),
        "cmax_median_B": float(np.median(b["cmax"])),
    }
    if avg_a is None:
        avg_a = run_dir / "ChanA" / "mean_unregistered.npy"
    if avg_b is None:
        avg_b = run_dir / "ChanB" / "mean_unregistered.npy"
    avg_a_img = _load_image(avg_a) if avg_a is not None and Path(avg_a).exists() else None
    avg_b_img = _load_image(avg_b) if avg_b is not None and Path(avg_b).exists() else None

    stats.update(_score_channel("A", avg_a_img, a["mean"], run_dir, signature_a))
    stats.update(_score_channel("B", avg_b_img, b["mean"], run_dir, signature_b))
    # aliases used by older notes / plots
    stats["ridge_reg_A"] = stats.get("ridge_reg_A")
    stats["ridge_reg_B"] = stats.get("ridge_reg_B")
    if "ridge_unreg_A" in stats:
        stats["ridge_avg_A"] = stats["ridge_unreg_A"]
    if "ridge_unreg_B" in stats:
        stats["ridge_avg_B"] = stats["ridge_unreg_B"]

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

    def _r(key, fmt=".3f"):
        v = stats.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        if isinstance(v, str):
            return v
        return format(v, fmt)

    lines = [
        f"xoff r = {stats['pearson_xoff']:.3f}",
        f"yoff r = {stats['pearson_yoff']:.3f}   (fringe-lock ~0.07)",
        f"cmax A/B med = {stats['cmax_median_A']:.4f} / {stats['cmax_median_B']:.4f}",
        "",
        "cell power ratio (reg/unreg)  A/B = "
        f"{_r('cell_power_ratio_A')} / {_r('cell_power_ratio_B')}",
        "fringe power ratio (reg/unreg) A/B = "
        f"{_r('fringe_power_ratio_A')} / {_r('fringe_power_ratio_B')}",
        f"verdict A/B = {_r('verdict_A')} / {_r('verdict_B')}",
        "",
        "legacy |ky|>0.05 ridge reg A/B = "
        f"{_r('ridge_reg_A', '.4f')} / {_r('ridge_reg_B', '.4f')}",
    ]
    if stats.get("ridge_unreg_A") is not None:
        lines.append(
            "legacy ridge unreg A/B = "
            f"{_r('ridge_unreg_A', '.4f')} / {_r('ridge_unreg_B', '.4f')}"
        )
    lines.append("Pass: cell ratio >1, fringe ratio ~<=1.")
    axes[1, 2].text(0.0, 0.52, "\n".join(lines), va="center", family="monospace", fontsize=8)
    axes[1, 2].set_xlim(0, 1)

    def _show(ax, img, title):
        if img is None:
            ax.axis("off")
            return
        lo, hi = np.percentile(img, (1, 99))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")

    def _baseline_title(path, letter):
        if path is None:
            return f"unregistered mean {letter}"
        name = Path(path).name.lower()
        if "stk_avg" in name:
            return f"stk_avg {letter}"
        return f"unregistered mean {letter}"

    _show(axes[2, 0], avg_a_img, _baseline_title(avg_a, "A"))
    _show(axes[2, 1], a["mean"], "registered mean A")
    _show(axes[2, 2], avg_b_img, _baseline_title(avg_b, "B"))
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
    p.add_argument("--signature-a", default=None)
    p.add_argument("--signature-b", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    compare(
        args.run_dir,
        avg_a=args.avg_a,
        avg_b=args.avg_b,
        out_path=args.out,
        signature_a=args.signature_a,
        signature_b=args.signature_b,
    )


if __name__ == "__main__":
    main()
