#!/usr/bin/env python3
"""Fair full-stack v2.1 tests: cell-ops MC, then CellPose means.

    python lab/pipeline/run_v21_fair_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.pipeline import run_mc_v21_bakeoff, run_seg_cellpose_compare


def main():
    overwrite = "--overwrite" in sys.argv
    argv = ["--overwrite"] if overwrite else []
    print("======== 1/2  MC on defringed_v21 ========")
    sys.argv = [run_mc_v21_bakeoff.__file__, *argv]
    err = run_mc_v21_bakeoff.main()
    if err:
        return err
    print("\n======== 2/2  CellPose raw vs v21 means ========")
    sys.argv = [run_seg_cellpose_compare.__file__]
    return run_seg_cellpose_compare.main()


if __name__ == "__main__":
    sys.exit(main())
