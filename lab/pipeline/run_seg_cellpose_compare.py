#!/usr/bin/env python3
"""CellPose (cyto3) on unregistered means: full raw vs full v2.1.

2D mask peek only — not GUI-evaluable. Forward comparisons use
run_seg_eval.py (detection + F/Fneu into suite2p/plane0).

Skips motion correction so the comparison is defringe vs raw anatomy.
ChanB is a wrong-prior control (stock Cellpose is soma-trained).

    seg_runs/cellpose_full/raw/ChanA|B
    seg_runs/cellpose_full/v21/ChanA|B
    seg_runs/cellpose_full/compare.png
    seg_runs/cellpose_full/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
RAW = {
    "A": SANDBOX / "inputs" / "raw" / "ChanA" / "ChanA_stk.tif",
    "B": SANDBOX / "inputs" / "raw" / "ChanB" / "ChanB_stk.tif",
}
V21 = {
    "A": SANDBOX / "inputs" / "defringed_v21" / "ChanA" / "ChanA_stk_defringed_v21.tif",
    "B": SANDBOX / "inputs" / "defringed_v21" / "ChanB" / "ChanB_stk_defringed_v21.tif",
}
OUT = SANDBOX / "seg_runs" / "cellpose_full"


def stack_mean_max(path):
    print(f"  loading {path}")
    mov = imread(str(path), is_ome=False)
    if mov.ndim != 3:
        raise ValueError(f"expected TZ/YX stack, got {mov.shape} from {path}")
    mean = mov.mean(axis=0).astype(np.float32)
    mx = mov.max(axis=0).astype(np.float32)
    del mov
    return mean, mx


def mask_stats(masks, um_per_px=478.69 / 512.0):
    masks = np.asarray(masks)
    n = int(masks.max())
    occupied = masks > 0
    coverage = float(occupied.mean())
    if n == 0:
        return {
            "n_roi": 0,
            "coverage": 0.0,
            "area_px_median": 0.0,
            "area_um2_median": 0.0,
            "eccentricity_median": float("nan"),
        }
    areas = np.bincount(masks.ravel())[1:].astype(np.float64)
    ecc = []
    try:
        from skimage.measure import regionprops
        for p in regionprops(masks):
            ecc.append(p.eccentricity)
    except Exception:
        ecc = []
    return {
        "n_roi": n,
        "coverage": coverage,
        "area_px_median": float(np.median(areas)),
        "area_um2_median": float(np.median(areas) * um_per_px ** 2),
        "eccentricity_median": float(np.median(ecc)) if ecc else float("nan"),
    }


def overlap_frac(a, b):
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    both = np.logical_and(a, b).sum()
    denom = np.logical_or(a, b).sum()
    return float(both / (denom + 1e-12))


def overlay(ax, img, masks, title):
    lo, hi = np.percentile(img, (1, 99))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    if int(np.max(masks)) > 0:
        from skimage.segmentation import find_boundaries
        bound = find_boundaries(masks, mode="outer")
        overlay_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        overlay_rgba[bound] = (1.0, 0.2, 0.2, 0.85)
        ax.imshow(overlay_rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def run_cellpose(img, model):
    img = np.asarray(img, dtype=np.float32)
    out = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        normalize=True,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    )
    return np.asarray(out[0], dtype=np.int32)


def load_model():
    from cellpose import models

    for name in ("cyto3", "cyto"):
        try:
            print(f"  loading Cellpose model_type={name} (CPU)")
            return models.CellposeModel(gpu=False, model_type=name), name
        except Exception as exc:
            print(f"  {name} failed: {exc}")
    raise RuntimeError("Could not load Cellpose cyto3 or cyto")


def save_channel(out_dir, mean, mx, masks_mean, stats, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mean.npy", mean)
    np.save(out_dir / "max.npy", mx)
    np.save(out_dir / "masks_mean.npy", masks_mean)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    overlay(ax, mean, masks_mean, f"{out_dir.name}  {model_name}  n={stats['n_roi']}")
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_mean.png", dpi=140)
    plt.close(fig)


def main():
    for p in list(RAW.values()) + list(V21.values()):
        if not Path(p).exists():
            print(f"ERROR: missing {p}")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    model, model_name = load_model()
    results = {"model": model_name, "gpu": False, "image": "unregistered mean of full stack"}
    masks_store = {}
    means = {}

    jobs = (
        ("raw", "A", RAW["A"]),
        ("raw", "B", RAW["B"]),
        ("v21", "A", V21["A"]),
        ("v21", "B", V21["B"]),
    )
    for kind, letter, path in jobs:
        print(f"\n=== {kind} Chan{letter} ===")
        mean, mx = stack_mean_max(path)
        print("  running Cellpose on mean")
        masks = run_cellpose(mean, model)
        stats = mask_stats(masks)
        stats["model"] = model_name
        stats["nframes"] = 5400
        print(f"  n={stats['n_roi']}  coverage={stats['coverage']:.3f}  "
              f"med area px={stats['area_px_median']:.1f}  "
              f"ecc={stats['eccentricity_median']:.3f}")
        out_dir = OUT / kind / f"Chan{letter}"
        save_channel(out_dir, mean, mx, masks, stats, model_name)
        results[f"{kind}_{letter}"] = stats
        masks_store[f"{kind}_{letter}"] = masks
        means[f"{kind}_{letter}"] = mean

    results["overlap_raw_AB"] = overlap_frac(masks_store["raw_A"], masks_store["raw_B"])
    results["overlap_v21_AB"] = overlap_frac(masks_store["v21_A"], masks_store["v21_B"])
    print(f"\nA vs B mask overlap raw={results['overlap_raw_AB']:.3f}  "
          f"v21={results['overlap_v21_AB']:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    overlay(axes[0, 0], means["raw_A"], np.zeros_like(masks_store["raw_A"]), "raw mean A")
    overlay(axes[0, 1], means["raw_A"], masks_store["raw_A"],
            f"raw A {model_name} n={results['raw_A']['n_roi']}")
    overlay(axes[0, 2], means["raw_B"], np.zeros_like(masks_store["raw_B"]), "raw mean B")
    overlay(axes[0, 3], means["raw_B"], masks_store["raw_B"],
            f"raw B {model_name} n={results['raw_B']['n_roi']} (wrong prior)")
    overlay(axes[1, 0], means["v21_A"], np.zeros_like(masks_store["v21_A"]), "v21 mean A")
    overlay(axes[1, 1], means["v21_A"], masks_store["v21_A"],
            f"v21 A {model_name} n={results['v21_A']['n_roi']}")
    overlay(axes[1, 2], means["v21_B"], np.zeros_like(masks_store["v21_B"]), "v21 mean B")
    overlay(axes[1, 3], means["v21_B"], masks_store["v21_B"],
            f"v21 B {model_name} n={results['v21_B']['n_roi']} (wrong prior)")
    fig.suptitle(
        "CellPose on unregistered means  |  ChanB = soma model on astrocytes (control)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=140)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {OUT / 'compare.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
