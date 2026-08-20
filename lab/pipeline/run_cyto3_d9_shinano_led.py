#!/usr/bin/env python3
"""cyto3 anatomical (mean) + F/Fneu on the five share-A Shinano LED FOVs.

Uses locked ANATOMICAL_BY_CELL_TYPE['astrocyte'] converted from Experiment.xml
(diameter_um → px, fs = frameRate/averageNum). Reuses ChanB data.bin.
OASIS off.

    python lab/pipeline/run_cyto3_d9_shinano_led.py
    python lab/pipeline/run_cyto3_d9_shinano_led.py --overwrite
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
from lab.pipeline.run_mc_shinano_led_batch import MC_ROOT, SESSIONS
from lab.pipeline.run_seg_eval import (
    clone_registered_plane,
    load_plane_view,
    overlay_rois,
    plane0_dir,
    plane0_is_complete,
)

OUT = MC_ROOT / "seg_cyto3_d9"
FS_HZ = 14.80
DIAMETER = 9
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
# Labeling split (not used for detection; titles only).
TRAIN = {"LED_x15_Level3b", "LED_x15_Level3"}
HOLDOUT = {"LED_x15_Level1", "LED_x15_Level5b", "LED_x15_Level5_001"}


def src_plane(name: str) -> Path:
    return MC_ROOT / name / "ChanB" / "suite2p" / "plane0"


def save_path0(name: str) -> Path:
    return OUT / name / "ChanB"


def short_name(name: str) -> str:
    return name.replace("LED_x15_", "")


def split_tag(name: str) -> str:
    if name in TRAIN:
        return "train"
    if name in HOLDOUT:
        return "holdout"
    return ""


def run_fov(name: str, overwrite: bool) -> Path:
    src = src_plane(name)
    dest = save_path0(name)
    plane = plane0_dir(dest)
    if not (src / "data.bin").exists():
        raise FileNotFoundError(f"missing registered binary {src / 'data.bin'}")
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    print(f"\n======== {name} ChanB cyto3 d={DIAMETER} -> {dest} ========")
    clone_registered_plane(src, dest)
    ops = apply_s2p_ops(
        default_ops(), tif_path=None, output_dir=dest, start=src, cell_type="astrocyte"
    )
    apply_seg_eval_ops(
        ops, method="cyto3", cell_type="astrocyte", channel_letter="B", start=src
    )
    ops["do_registration"] = 0
    ops["roidetect"] = True
    ops["spikedetect"] = False
    run_s2p(
        ops=ops,
        db={
            "save_path0": str(dest),
            "save_folder": SEG_EVAL["save_folder"],
            "fast_disk": str(dest),
            "data_path": [str(src)],
        },
    )
    missing = [n for n in SEG_EVAL["plane0_required"] if not (plane / n).exists()]
    if missing:
        raise FileNotFoundError(f"{plane} missing {missing}")
    print(f"  GUI: suite2p -> {plane / 'stat.npy'}")
    print(f"  GUI: s2p_Trace_Curation -> {plane.parent}")
    return plane


def write_compare() -> Path:
    views = []
    metrics = {
        "channel": "B",
        "cell_type": "astrocyte",
        "method": "cyto3",
        "anatomical_only": 2,
        "diameter": DIAMETER,
        "flow_threshold": FLOW_THRESHOLD,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "spatial_hp_cp": 0,
        "spikedetect": False,
        "fs": FS_HZ,
        "conditions": {},
    }
    for name in SESSIONS:
        plane = plane0_dir(save_path0(name))
        view = load_plane_view(plane)
        if view is None:
            raise FileNotFoundError(f"incomplete {plane}")
        views.append((name, view))
        metrics["conditions"][name] = {
            "n_roi": view["n_roi"],
            "coverage": view["coverage"],
            "split": split_tag(name),
            "plane0": str(plane),
        }

    n = len(views)
    fig = plt.figure(figsize=(3.4 * n, 3.8), layout="constrained")
    gs = GridSpec(1, n, figure=fig)
    for i, (name, view) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        tag = split_tag(name)
        title = f"{short_name(name)}  n={view['n_roi']}"
        if tag:
            title += f"  ({tag})"
        overlay_rois(ax, view["mean"], view["masks"], title)
    fig.suptitle(
        "ChanB  |  share-A  |  cyto3  anatomical_only=2  "
        f"diameter={DIAMETER}  flow={FLOW_THRESHOLD}  cellprob={CELLPROB_THRESHOLD}",
        fontsize=11,
    )
    out = OUT / "compare.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    return out


def main():
    overwrite = "--overwrite" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SESSIONS:
        run_fov(name, overwrite)
    write_compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
