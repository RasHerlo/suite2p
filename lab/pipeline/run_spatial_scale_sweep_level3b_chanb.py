#!/usr/bin/env python3
"""Temporal spatial_scale sweep on share-A Level3b ChanB (G-Flamp).

tau locked at 1 s. OASIS off. Reuses the registered data.bin.

    python lab/pipeline/run_spatial_scale_sweep_level3b_chanb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from suite2p import default_ops
from suite2p.run_s2p import run_s2p

from lab.configs.defaults import SEG_EVAL, apply_s2p_ops, apply_seg_eval_ops
from lab.pipeline.run_seg_eval import (
    clone_registered_plane,
    draw_condition_row,
    load_plane_view,
    plane0_dir,
    plane0_is_complete,
    write_condition_overview,
)
from lab.pipeline.run_tau_sweep_level3b_chanb import (
    FS_HZ,
    HIGH_PASS_SECONDS,
    SRC_PLANE,
    high_pass_bins,
    write_empty_plane,
)

MC = SRC_PLANE.parents[2]
OUT = MC / "seg_spatial_scale_sweep"
TAU = 1.0
# suite2p: 1=6 px, 2=12 px, 3=24 px (4=48 px left for a later pass)
SCALES = (1, 2, 3)
SCALE_PX = {1: 6, 2: 12, 3: 24, 4: 48}


def scale_dir(scale: int) -> Path:
    return OUT / f"scale{scale}" / "ChanB"


def run_scale(scale: int, overwrite: bool) -> Path:
    save_path0 = scale_dir(scale)
    plane = plane0_dir(save_path0)
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    print(
        f"\n======== ChanB temporal tau={TAU:g} spatial_scale={scale} "
        f"({SCALE_PX[scale]} px) -> {save_path0} ========"
    )
    clone_registered_plane(SRC_PLANE, save_path0)
    ops = apply_s2p_ops(
        default_ops(), tif_path=None, output_dir=save_path0, start=SRC_PLANE
    )
    apply_seg_eval_ops(
        ops, method="temporal", channel_letter="B", start=SRC_PLANE
    )
    ops["tau"] = TAU
    ops["high_pass"] = high_pass_bins(TAU)
    ops["spatial_scale"] = int(scale)
    ops["do_registration"] = 0
    ops["roidetect"] = True
    ops["anatomical_only"] = 0
    try:
        run_s2p(
            ops=ops,
            db={
                "save_path0": str(save_path0),
                "save_folder": SEG_EVAL["save_folder"],
                "fast_disk": str(save_path0),
                "data_path": [str(SRC_PLANE)],
            },
        )
    except ValueError as exc:
        if "no ROIs were found" not in str(exc):
            raise
        print(f"  detect found 0 ROIs at spatial_scale={scale}")
        write_empty_plane(plane)
    missing = [n for n in SEG_EVAL["plane0_required"] if not (plane / n).exists()]
    if missing:
        raise FileNotFoundError(f"{plane} missing {missing}")
    print(f"  GUI: suite2p -> {plane / 'stat.npy'}")
    print(f"  GUI: s2p_Trace_Curation -> {plane.parent}")
    return plane


def write_compare() -> Path:
    views = []
    metrics = {
        "fs": FS_HZ,
        "channel": "B",
        "method": "temporal",
        "tau": TAU,
        "spatial_scales": [],
        "conditions": {},
    }
    for scale in SCALES:
        save_path0 = scale_dir(scale)
        plane = plane0_dir(save_path0)
        view = load_plane_view(plane)
        if view is None:
            raise FileNotFoundError(f"incomplete {plane}")
        px = SCALE_PX[scale]
        label = (
            f"scale={scale} ({px} px)  n={view['n_roi']}  cov={view['coverage']:.3f}"
        )
        write_condition_overview(save_path0, view, label)
        views.append((scale, label, view))
        metrics["spatial_scales"].append(scale)
        metrics["conditions"][f"scale{scale}"] = {
            "n_roi": view["n_roi"],
            "coverage": view["coverage"],
            "pixels": px,
            "tau": TAU,
            "high_pass_bins": high_pass_bins(TAU),
            "plane0": str(plane),
        }
    fig = plt.figure(figsize=(14, 3.6 * len(views)), layout="constrained")
    gs = GridSpec(len(views), 3, figure=fig, width_ratios=[1, 1, 2.2])
    for i, (_scale, label, view) in enumerate(views):
        draw_condition_row(fig, gs, i, 0, view, label)
    fig.suptitle(
        f"Level3b ChanB  |  share-A v22  |  temporal spatial_scale sweep  |  tau={TAU:g}s  |  mean  |  ROIs  |  F raster",
        fontsize=11,
    )
    out = OUT / "compare.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    return out


def main():
    overwrite = "--overwrite" in sys.argv
    if not (SRC_PLANE / "data.bin").exists():
        print(f"ERROR: missing {SRC_PLANE / 'data.bin'}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for scale in SCALES:
        run_scale(scale, overwrite)
    write_compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
