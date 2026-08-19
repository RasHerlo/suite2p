#!/usr/bin/env python3
"""Segmentation + extraction bakeoff → GUI-openable suite2p/plane0 folders.

Each arm runs ROI detection and extracts F and Fneu. OASIS is off
(ops['spikedetect']=False); suite2p still writes zeros to spks.npy so the
suite2p GUI loader accepts the folder.

    python lab/pipeline/run_seg_eval.py
    python lab/pipeline/run_seg_eval.py --methods temporal,cyto3
    python lab/pipeline/run_seg_eval.py --overwrite

Layout (sandbox):

    seg_runs/v21_cell_<method>/ChanA|B/suite2p/plane0/
    seg_runs/v21_cell_eval/compare.png
        (registered mean | mean+ROIs | F raster, one row per condition)

Open:
    suite2p GUI          → plane0/stat.npy
    s2p_Trace_Curation   → the suite2p folder (parent of plane0)

Does not apply lab ellipticity ROI filtering. Does not extract paper traces.
ChanB + cyto3 remains a wrong-prior control until an astrocyte model exists.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from suite2p import default_ops
from suite2p.run_s2p import run_s2p

from lab.configs.defaults import (
    SEG_EVAL,
    apply_s2p_ops,
    apply_seg_eval_ops,
)

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
V21 = SANDBOX / "inputs" / "defringed_v21"
OUT = SANDBOX / "seg_runs"
FS_HZ = 14.80  # Experiment.xml frameRate/averageNum
TIFFS = {
    "A": V21 / "ChanA" / "ChanA_stk_defringed_v21.tif",
    "B": V21 / "ChanB" / "ChanB_stk_defringed_v21.tif",
}


def plane0_dir(save_path0: Path) -> Path:
    return save_path0 / SEG_EVAL["save_folder"] / "plane0"


def plane0_is_complete(plane: Path) -> bool:
    return all((plane / name).exists() for name in SEG_EVAL["plane0_required"])


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ops_for(tif_path: Path, save_path0: Path, method: str, *, detect: bool) -> dict:
    ops = apply_s2p_ops(default_ops(), tif_path=tif_path, output_dir=save_path0)
    apply_seg_eval_ops(ops, method=method)
    ops["fs"] = FS_HZ
    ops["roidetect"] = detect
    ops["tiff_list"] = [tif_path.name]
    return ops


def _db(tif_path: Path, save_path0: Path) -> dict:
    return {
        "data_path": [str(tif_path.parent)],
        "tiff_list": [tif_path.name],
        "save_path0": str(save_path0),
        "save_folder": SEG_EVAL["save_folder"],
        "fast_disk": str(save_path0),
    }


def ensure_registered(letter: str, overwrite: bool) -> Path:
    """Write one registered data.bin per channel (detection off)."""
    tif = TIFFS[letter]
    save_path0 = OUT / "_bin" / "v21_cell" / f"Chan{letter}"
    plane = plane0_dir(save_path0)
    bin_path = plane / "data.bin"
    if bin_path.exists() and not overwrite:
        print(f"  reuse registered binary {bin_path}")
        return plane
    if overwrite and save_path0.exists():
        shutil.rmtree(save_path0)
    print(f"\n======== register Chan{letter} -> {save_path0} ========")
    ops = _ops_for(tif, save_path0, "temporal", detect=False)
    run_s2p(ops=ops, db=_db(tif, save_path0))
    if not bin_path.exists():
        raise FileNotFoundError(f"registration did not write {bin_path}")
    return plane


def clone_registered_plane(src_plane: Path, dest_save0: Path) -> Path:
    dest_plane = plane0_dir(dest_save0)
    dest_plane.mkdir(parents=True, exist_ok=True)
    _hardlink_or_copy(src_plane / "data.bin", dest_plane / "data.bin")
    shutil.copy2(src_plane / "ops.npy", dest_plane / "ops.npy")
    for name in ("stat.npy", "F.npy", "Fneu.npy", "iscell.npy", "spks.npy"):
        leftover = dest_plane / name
        if leftover.exists():
            leftover.unlink()
    return dest_plane


def run_method(letter: str, method: str, src_plane: Path, overwrite: bool) -> Path:
    tif = TIFFS[letter]
    save_path0 = OUT / f"v21_cell_{method}" / f"Chan{letter}"
    plane = plane0_dir(save_path0)
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    if overwrite and save_path0.exists():
        shutil.rmtree(save_path0)
    print(f"\n======== {method} Chan{letter} -> {save_path0} ========")
    clone_registered_plane(src_plane, save_path0)
    ops = _ops_for(tif, save_path0, method, detect=True)
    ops["do_registration"] = 0
    run_s2p(ops=ops, db=_db(tif, save_path0))
    missing = [n for n in SEG_EVAL["plane0_required"] if not (plane / n).exists()]
    if missing:
        raise FileNotFoundError(f"{plane} missing {missing}")
    print(f"  GUI: suite2p → {plane / 'stat.npy'}")
    print(f"  GUI: s2p_Trace_Curation → {plane.parent}")
    return plane


def masks_from_stat(stat, Ly: int, Lx: int) -> np.ndarray:
    masks = np.zeros((Ly, Lx), dtype=np.int32)
    for i, roi in enumerate(stat, start=1):
        ypix = np.asarray(roi["ypix"], dtype=np.int64)
        xpix = np.asarray(roi["xpix"], dtype=np.int64)
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        masks[ypix[valid], xpix[valid]] = i
    return masks


def zscore_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (arr - mu) / sd


def overlay_rois(ax, img, masks, title: str) -> None:
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if not np.isfinite(lo) or lo == hi:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img) + 1e-6)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    if masks is not None and int(np.max(masks)) > 0:
        from skimage.segmentation import find_boundaries

        bound = find_boundaries(masks, mode="outer")
        rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        rgba[bound] = (1.0, 0.2, 0.2, 0.85)
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_f_raster(ax, F: np.ndarray, fs: float, title: str):
    F = np.asarray(F)
    if F.ndim != 2 or F.shape[0] == 0:
        ax.set_axis_off()
        ax.set_title(title + "  (no ROIs)", fontsize=9)
        return None
    z = zscore_rows(F)
    n_roi, n_frames = z.shape
    t_end = n_frames / float(fs) if fs else n_frames
    vmax = np.percentile(np.abs(z), 99)
    vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else float(vmax)
    im = ax.imshow(
        z,
        aspect="auto",
        cmap="gray_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(0, t_end, n_roi, 0),
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("ROI", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def load_plane_view(plane: Path) -> dict | None:
    if not plane0_is_complete(plane):
        return None
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    F = np.load(plane / "F.npy")
    mean = np.asarray(ops["meanImg"], dtype=np.float32)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    fs = float(ops.get("fs", FS_HZ) or FS_HZ)
    return {
        "mean": mean,
        "masks": masks_from_stat(stat, Ly, Lx),
        "F": np.asarray(F),
        "n_roi": int(len(stat)),
        "fs": fs,
    }


def condition_title(method: str, letter: str, n_roi: int) -> str:
    extra = "  (wrong prior)" if method == "cyto3" and letter == "B" else ""
    return f"{method} Chan{letter}{extra}  n={n_roi}"


def draw_condition_row(fig, gs, row: int, view: dict, label: str):
    ax_mean = fig.add_subplot(gs[row, 0])
    ax_roi = fig.add_subplot(gs[row, 1])
    ax_ras = fig.add_subplot(gs[row, 2])
    overlay_rois(ax_mean, view["mean"], None, f"{label}  mean")
    overlay_rois(ax_roi, view["mean"], view["masks"], f"{label}  ROIs")
    im = plot_f_raster(ax_ras, view["F"], view["fs"], f"{label}  F raster (z / ROI)")
    if im is not None:
        cb = fig.colorbar(im, ax=ax_ras, fraction=0.046, pad=0.02)
        cb.set_label("z", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return ax_mean, ax_roi, ax_ras


def write_condition_overview(save_path0: Path, view: dict, label: str) -> Path:
    fig = plt.figure(figsize=(14, 4.2), layout="constrained")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 2.2])
    draw_condition_row(fig, gs, 0, view, label)
    out = save_path0 / "overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def write_compare_figure(methods: list[str]) -> Path | None:
    rows = []
    for method in methods:
        for letter in ("A", "B"):
            save_path0 = OUT / f"v21_cell_{method}" / f"Chan{letter}"
            plane = plane0_dir(save_path0)
            view = load_plane_view(plane)
            if view is None:
                print(f"  skip figure row, incomplete {plane}")
                continue
            label = condition_title(method, letter, view["n_roi"])
            write_condition_overview(save_path0, view, label)
            rows.append((label, view))
    if not rows:
        print("  no complete planes; skip compare.png")
        return None
    fig = plt.figure(figsize=(14, 3.6 * len(rows)), layout="constrained")
    gs = GridSpec(len(rows), 3, figure=fig, width_ratios=[1, 1, 2.2])
    for i, (label, view) in enumerate(rows):
        draw_condition_row(fig, gs, i, view, label)
    fig.suptitle(
        "v21 cell-ops  |  registered mean  |  ROIs  |  F raster (all detected ROIs)",
        fontsize=11,
    )
    out_dir = OUT / "v21_cell_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "compare.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def parse_methods(argv: list[str]) -> list[str]:
    methods = list(SEG_EVAL["methods"])
    for i, arg in enumerate(argv):
        if arg == "--methods" and i + 1 < len(argv):
            methods = [m.strip() for m in argv[i + 1].split(",") if m.strip()]
    unknown = [m for m in methods if m not in SEG_EVAL["methods"]]
    if unknown:
        raise ValueError(
            f"unknown methods {unknown}; expected {tuple(SEG_EVAL['methods'])}"
        )
    return methods


def main() -> int:
    overwrite = "--overwrite" in sys.argv
    methods = parse_methods(sys.argv)
    for letter, path in TIFFS.items():
        if not path.exists():
            print(f"ERROR: missing {path}")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for letter in ("A", "B"):
        src = ensure_registered(letter, overwrite)
        for method in methods:
            run_method(letter, method, src, overwrite)
    print("\n======== compare.png ========")
    write_compare_figure(methods)
    print("\nseg eval done. Open each Chan*/suite2p folder in either GUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
