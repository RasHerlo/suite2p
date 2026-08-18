#!/usr/bin/env python3
"""Independent raw ChanA/B motion-correction bakeoff (legacy vs cell-ops).

Writes under the Level3b copy sandbox:

    mc_runs/raw_legacy/ChanA|B
    mc_runs/raw_cell/ChanA|B
    mc_runs/raw_legacy/compare_AB.png
    mc_runs/raw_cell/compare_AB.png

Does not share shifts across channels. Does not write full registered TIFFs.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.configs.defaults import REGISTRATION, REGISTRATION_LEGACY
from lab.pipeline.compare_mc_channels import compare
from lab.pipeline.fringe_robust_register import process_tree

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
RAW = SANDBOX / "inputs" / "raw"
MC = SANDBOX / "mc_runs"


def _run(name, cfg_base, overwrite):
    cfg = deepcopy(cfg_base)
    cfg["share_shifts_across_channels"] = False
    cfg["write_registered_tif"] = False
    out = MC / name
    print(f"\n======== {name} -> {out} ========")
    process_tree(
        RAW,
        cfg,
        share_shifts=False,
        overwrite=overwrite,
        output_root=out,
    )
    compare(
        out,
        avg_a=RAW / "ChanA" / "ChanA_stk_avg.tif",
        avg_b=RAW / "ChanB" / "ChanB_stk_avg.tif",
        out_path=out / "compare_AB.png",
    )


def main():
    overwrite = "--overwrite" in sys.argv
    if not RAW.exists():
        print(f"ERROR: missing sandbox raw inputs at {RAW}")
        return 1
    MC.mkdir(parents=True, exist_ok=True)
    _run("raw_legacy", REGISTRATION_LEGACY, overwrite)
    _run("raw_cell", REGISTRATION, overwrite)
    print("\nBakeoff done.")
    print(f"  {MC / 'raw_legacy' / 'compare_AB.png'}")
    print(f"  {MC / 'raw_cell' / 'compare_AB.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
