#!/usr/bin/env python3
"""Locked default detection + F/Fneu after share-A MC.

Runs **temporal** and **anatomical (cyto3)** on each PMT. Same registered
``data.bin`` per FOV×channel; two ``plane0`` folders. Intercalation is
s2p_Trace_Curation, not this repo.

Knobs follow cell type and Experiment.xml (fs, µm/px). Do not hardcode
diameter=9 or spatial_scale=1 here.

    python lab/pipeline/run_seg_locked.py
    python lab/pipeline/run_seg_locked.py --overwrite
    python lab/pipeline/run_seg_locked.py --methods temporal,cyto3 --letters B

Sidecar::

    mc_runs/.../seg_locked/<FOV>/Chan{A|B}_{temporal|cyto3}/suite2p/plane0/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from suite2p import default_ops
from suite2p.run_s2p import run_s2p

from lab.configs.defaults import SEG_EVAL, apply_s2p_ops, apply_seg_eval_ops
from lab.pipeline.run_mc_shinano_led_batch import MC_ROOT, SESSIONS
from lab.pipeline.run_seg_eval import (
    clone_registered_plane,
    load_plane_view,
    plane0_dir,
    plane0_is_complete,
)
from lab.pipeline.run_tau_sweep_level3b_chanb import write_empty_plane

OUT = MC_ROOT / "seg_locked"
METHODS = tuple(SEG_EVAL["methods"])
LETTERS = ("A", "B")


def src_plane(name: str, letter: str) -> Path:
    return MC_ROOT / name / f"Chan{letter}" / "suite2p" / "plane0"


def save_path0(name: str, letter: str, method: str) -> Path:
    return OUT / name / f"Chan{letter}_{method}"


def run_one(name: str, letter: str, method: str, overwrite: bool) -> Path:
    src = src_plane(name, letter)
    dest = save_path0(name, letter, method)
    plane = plane0_dir(dest)
    if not (src / "data.bin").exists():
        raise FileNotFoundError(f"missing registered binary {src / 'data.bin'}")
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    print(f"\n======== {name} Chan{letter} {method} -> {dest} ========")
    clone_registered_plane(src, dest)
    ops = apply_s2p_ops(
        default_ops(), tif_path=None, output_dir=dest, start=src
    )
    apply_seg_eval_ops(
        ops,
        method=method,
        channel_letter=letter,
        computer=ops.get("computer"),
        start=src,
    )
    ops["do_registration"] = 0
    ops["roidetect"] = True
    ops["spikedetect"] = False
    try:
        run_s2p(
            ops=ops,
            db={
                "save_path0": str(dest),
                "save_folder": SEG_EVAL["save_folder"],
                "fast_disk": str(dest),
                "data_path": [str(src)],
            },
        )
    except ValueError as exc:
        if "no ROIs were found" not in str(exc):
            raise
        print(f"  detect found 0 ROIs")
        write_empty_plane(plane)
    missing = [n for n in SEG_EVAL["plane0_required"] if not (plane / n).exists()]
    if missing:
        raise FileNotFoundError(f"{plane} missing {missing}")
    print(f"  GUI: suite2p -> {plane / 'stat.npy'}")
    print(f"  GUI: s2p_Trace_Curation -> {plane.parent}")
    return plane


def write_metrics(methods, letters) -> Path:
    metrics = {"root": str(OUT), "methods": list(methods), "letters": list(letters),
               "conditions": {}}
    for name in SESSIONS:
        for letter in letters:
            for method in methods:
                plane = plane0_dir(save_path0(name, letter, method))
                view = load_plane_view(plane)
                key = f"{name}_Chan{letter}_{method}"
                if view is None:
                    metrics["conditions"][key] = {"status": "incomplete", "plane0": str(plane)}
                    continue
                metrics["conditions"][key] = {
                    "n_roi": view["n_roi"],
                    "coverage": view["coverage"],
                    "fs": view["fs"],
                    "plane0": str(plane),
                }
    path = OUT / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {path}")
    return path


def _csv(argv, flag, default):
    values = list(default)
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            values = [x.strip() for x in argv[i + 1].split(",") if x.strip()]
    return values


def main():
    overwrite = "--overwrite" in sys.argv
    methods = _csv(sys.argv, "--methods", METHODS)
    letters = tuple(x.upper() for x in _csv(sys.argv, "--letters", LETTERS))
    unknown = [m for m in methods if m not in SEG_EVAL["methods"]]
    if unknown:
        raise ValueError(f"unknown methods {unknown}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SESSIONS:
        for letter in letters:
            for method in methods:
                run_one(name, letter, method, overwrite)
    write_metrics(methods, letters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
