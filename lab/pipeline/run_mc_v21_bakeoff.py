#!/usr/bin/env python3
"""Independent + share-A cell-ops MC on full-stack v2.1 defringe.

Writes:
    mc_runs/v21_cell/ChanA|B
    mc_runs/v21_cell_shareA/ChanA|B
    compare_AB.png in each

Ridge baseline is the unregistered mean of the *defringed* movie
(mean_unregistered.npy), not raw stk_avg.

Does not write full registered TIFFs. Does not touch original DATA.
"""

from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.configs.defaults import REGISTRATION
from lab.pipeline.compare_mc_channels import compare
from lab.pipeline.fringe_robust_register import process_tree

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
V21 = SANDBOX / "inputs" / "defringed_v21"
MC = SANDBOX / "mc_runs"


def _cfg():
    cfg = deepcopy(REGISTRATION)
    cfg["share_shifts_across_channels"] = False
    cfg["write_registered_tif"] = False
    return cfg


def _compare(out):
    compare(out, out_path=out / "compare_AB.png")


def run_independent(overwrite):
    out = MC / "v21_cell"
    print(f"\n======== v21_cell -> {out} ========")
    process_tree(
        V21,
        _cfg(),
        share_shifts=False,
        overwrite=overwrite,
        output_root=out,
    )
    _compare(out)


def run_share_a(overwrite):
    src_a = MC / "v21_cell" / "ChanA"
    out = MC / "v21_cell_shareA"
    if not (src_a / "offsets.npz").exists():
        print(f"ERROR: missing {src_a / 'offsets.npz'}")
        return 1
    dst_a = out / "ChanA"
    dst_b = out / "ChanB"
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if overwrite or not (dst_a / "offsets.npz").exists():
        if dst_a.exists():
            shutil.rmtree(dst_a)
        shutil.copytree(src_a, dst_a)
        print(f"copied ChanA products -> {dst_a}")
    if overwrite and dst_b.exists():
        shutil.rmtree(dst_b)
    cfg = _cfg()
    cfg["share_shifts_across_channels"] = True
    print(f"\n======== v21_cell_shareA -> {out} ========")
    print("ChanA shifts reused; ChanB is apply-only")
    process_tree(
        V21,
        cfg,
        share_shifts=True,
        overwrite=False,
        output_root=out,
    )
    _compare(out)
    return 0


def main():
    overwrite = "--overwrite" in sys.argv
    tif_a = V21 / "ChanA" / "ChanA_stk_defringed_v21.tif"
    tif_b = V21 / "ChanB" / "ChanB_stk_defringed_v21.tif"
    if not tif_a.exists() or not tif_b.exists():
        print(f"ERROR: missing v2.1 stacks at {V21}")
        return 1
    MC.mkdir(parents=True, exist_ok=True)
    run_independent(overwrite)
    err = run_share_a(overwrite)
    if err:
        return err
    print("\nv21 MC bakeoff done.")
    print(f"  {MC / 'v21_cell' / 'compare_AB.png'}")
    print(f"  {MC / 'v21_cell_shareA' / 'compare_AB.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
