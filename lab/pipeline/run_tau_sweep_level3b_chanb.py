#!/usr/bin/env python3
"""Temporal tau sweep on share-A Level3b ChanB (G-Flamp).

Reuses the registered data.bin. OASIS off. Does not train Cellpose.

    python lab/pipeline/run_tau_sweep_level3b_chanb.py
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

MC = (
    Path(r"F:\bPACNewData2026")
    / "AC_cAMP_Neu_Ca_C1_C2"
    / "mc_runs"
    / "260511"
    / "C1_RLV_LW_maybe"
    / "LED_x15_Level3b"
)
SRC_PLANE = MC / "ChanB" / "suite2p" / "plane0"
OUT = MC / "seg_tau_sweep"
TAUS = (1.0, 3.0, 5.0, 10.0)
FS_HZ = 14.80  # Experiment.xml frameRate/averageNum
# Lock scale so tau is the only detection knob. Auto-scale was 12 px at tau=1
# and 6 px at tau=3; at tau=5 auto failed and detection aborted.
SPATIAL_SCALE = 2  # 12 px
# Default high_pass=100 is in *binned* frames (~100 s at tau=1). Scale so the
# physical window stays ~100 s when tau increases (else tau=10 has 36 bins).
HIGH_PASS_SECONDS = 100.0


def tau_dir(tau: float) -> Path:
    tag = str(int(tau)) if float(tau).is_integer() else str(tau).replace(".", "p")
    return OUT / f"tau{tag}" / "ChanB"


def high_pass_bins(tau: float) -> int:
    return max(3, int(round(HIGH_PASS_SECONDS / float(tau))))


def write_empty_plane(plane: Path) -> None:
    """suite2p raises if sparse detect finds nothing; still write a GUI folder."""
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    nframes = int(ops["nframes"])
    np.save(plane / "stat.npy", np.array([], dtype=object))
    np.save(plane / "F.npy", np.zeros((0, nframes), np.float32))
    np.save(plane / "Fneu.npy", np.zeros((0, nframes), np.float32))
    np.save(plane / "iscell.npy", np.zeros((0, 2), np.float32))
    np.save(plane / "spks.npy", np.zeros((0, nframes), np.float32))
    print(f"  wrote empty plane0 (0 ROIs) at {plane}")


def run_tau(tau: float, overwrite: bool) -> Path:
    save_path0 = tau_dir(tau)
    plane = plane0_dir(save_path0)
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    print(f"\n======== ChanB temporal tau={tau:g} s -> {save_path0} ========")
    clone_registered_plane(SRC_PLANE, save_path0)
    ops = apply_s2p_ops(
        default_ops(), tif_path=None, output_dir=save_path0, start=SRC_PLANE
    )
    apply_seg_eval_ops(
        ops, method="temporal", channel_letter="B", start=SRC_PLANE
    )
    ops["tau"] = float(tau)
    ops["high_pass"] = high_pass_bins(tau)
    ops["spatial_scale"] = SPATIAL_SCALE
    ops["do_registration"] = 0
    ops["roidetect"] = True
    ops["anatomical_only"] = 0
    print(
        f"  bin_size ~ {int(round(tau * FS_HZ))} frames  "
        f"high_pass={ops['high_pass']} bins  spatial_scale={SPATIAL_SCALE}"
    )
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
        print(f"  detect found 0 ROIs at tau={tau:g}")
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
        "spatial_scale": SPATIAL_SCALE,
        "taus": [],
        "conditions": {},
    }
    for tau in TAUS:
        save_path0 = tau_dir(tau)
        plane = plane0_dir(save_path0)
        view = load_plane_view(plane)
        if view is None:
            raise FileNotFoundError(f"incomplete {plane}")
        hp = high_pass_bins(tau)
        label = f"tau={tau:g}s  n={view['n_roi']}  cov={view['coverage']:.3f}"
        write_condition_overview(save_path0, view, label)
        views.append((tau, label, view))
        metrics["taus"].append(tau)
        metrics["conditions"][f"tau{tau:g}"] = {
            "n_roi": view["n_roi"],
            "coverage": view["coverage"],
            "high_pass_bins": hp,
            "bin_frames": int(round(tau * FS_HZ)),
            "spatial_scale": SPATIAL_SCALE,
            "plane0": str(plane),
        }
    fig = plt.figure(figsize=(14, 3.6 * len(views)), layout="constrained")
    gs = GridSpec(len(views), 3, figure=fig, width_ratios=[1, 1, 2.2])
    for i, (_tau, label, view) in enumerate(views):
        draw_condition_row(fig, gs, i, 0, view, label)
    fig.suptitle(
        f"Level3b ChanB  |  share-A v22  |  temporal tau sweep  |  spatial_scale={SPATIAL_SCALE} (12 px)  |  mean  |  ROIs  |  F raster",
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
    for tau in TAUS:
        run_tau(tau, overwrite)
    write_compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
