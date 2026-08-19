#!/usr/bin/env python3
"""Two raw-MC follow-up checks on the Level3b copy sandbox.

1. raw_cell_shareA  -- apply existing cell-ops ChanA shifts to ChanB
   (do not re-estimate B). If A tracked cells, B Fourier-y ridge should
   fall vs independent raw_cell ChanB.

2. raw_cell_lowpass -- independent A/B, align_filter=lowpass (phasecorr
   weighting only; delivered intensities unchanged).

Does not write full registered TIFFs. Does not touch the original DATA tree.
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
RAW = SANDBOX / "inputs" / "raw"
MC = SANDBOX / "mc_runs"
CELL_A = MC / "raw_cell" / "ChanA"


def _compare(out):
    compare(out, out_path=out / "compare_AB.png")


def run_share_a(overwrite):
    """Reuse independent cell-ops ChanA traces; apply them to ChanB."""
    out = MC / "raw_cell_shareA"
    if not CELL_A.joinpath("offsets.npz").exists():
        print(f"ERROR: missing {CELL_A / 'offsets.npz'} (run raw_cell bakeoff first)")
        return 1
    dst_a = out / "ChanA"
    dst_b = out / "ChanB"
    out.mkdir(parents=True, exist_ok=True)
    if not (dst_a / "offsets.npz").exists():
        shutil.copytree(CELL_A, dst_a)
        print(f"copied ChanA products -> {dst_a}")
    if overwrite and dst_b.exists():
        shutil.rmtree(dst_b)
    cfg = deepcopy(REGISTRATION)
    cfg["share_shifts_across_channels"] = True
    cfg["write_registered_tif"] = False
    print(f"\n======== raw_cell_shareA -> {out} ========")
    print("ChanA shifts reused; ChanB is apply-only")
    process_tree(
        RAW,
        cfg,
        share_shifts=True,
        overwrite=False,
        output_root=out,
    )
    _compare(out)
    return 0


def run_lowpass(overwrite):
    cfg = deepcopy(REGISTRATION)
    cfg["share_shifts_across_channels"] = False
    cfg["write_registered_tif"] = False
    cfg["align_filter"] = "lowpass"
    cfg["lowpass_sigma"] = 4.0
    out = MC / "raw_cell_lowpass"
    print(f"\n======== raw_cell_lowpass -> {out} ========")
    process_tree(
        RAW,
        cfg,
        share_shifts=False,
        overwrite=overwrite,
        output_root=out,
    )
    _compare(out)
    return 0


def main():
    overwrite = "--overwrite" in sys.argv
    if not RAW.exists():
        print(f"ERROR: missing sandbox raw inputs at {RAW}")
        return 1
    MC.mkdir(parents=True, exist_ok=True)
    err = run_share_a(overwrite)
    if err:
        return err
    err = run_lowpass(overwrite)
    if err:
        return err
    print("\nFollow-up done.")
    print(f"  {MC / 'raw_cell_shareA' / 'compare_AB.png'}")
    print(f"  {MC / 'raw_cell_lowpass' / 'compare_AB.png'}")
    print("Judge shareA by ChanB ridge vs independent raw_cell (shifts match A by construction).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
