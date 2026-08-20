#!/usr/bin/env python3
"""Export share-A astrocyte meanImgs for Cellpose human-in-the-loop labeling.

    python lab/pipeline/prepare_cellpose_astro_labels.py
    python lab/pipeline/prepare_cellpose_astro_labels.py --gui   # open Level3b

Train FOVs: Level3b, Level3. Hold out Level1 (paper) and Level5*.
Label somata only. Do not paint residual fringes (see guides/).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tifffile import imwrite

MC_ROOT = (
    Path(r"F:\bPACNewData2026")
    / "AC_cAMP_Neu_Ca_C1_C2"
    / "mc_runs"
    / "260511"
    / "C1_RLV_LW_maybe"
)
OUT = MC_ROOT / "cellpose_astro_labels"
TRAIN = ("LED_x15_Level3b", "LED_x15_Level3")
HOLDOUT = ("LED_x15_Level1", "LED_x15_Level5b", "LED_x15_Level5_001")
README = """Astrocyte soma labels (Cellpose)

Target: compact G-Flamp somata on the share-A registered mean.
Not territories, not processes, not fringe stripes.

Train (label these first):
  train/LED_x15_Level3b_mean.tif   <-- start here
  train/LED_x15_Level3_mean.tif

Hold out (do not label until a round is trained):
  holdout/LED_x15_Level1_mean.tif     paper FOV
  holdout/LED_x15_Level5b_mean.tif
  holdout/LED_x15_Level5_001_mean.tif

Guides (do not draw on stripe-shaped bits):
  guides/<FOV>_independent_meanImg.png

In Cellpose GUI: model cyto3 as a *draft* to edit, not as GT.
Save masks in the same folder (File → Save). Cellpose writes *_seg.npy.

Weights later go in the suite2p repo models/ (gitignored) and
CELLPOSE['models']['astrocyte'].
"""


def _to_uint16(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99.9))
    if not np.isfinite(lo) or lo == hi:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img) + 1e-6)
    scaled = np.clip((img - lo) / (hi - lo), 0, 1) * 65535.0
    return scaled.astype(np.uint16)


def export_fov(name: str, dest_dir: Path) -> Path:
    ops_path = MC_ROOT / name / "ChanB" / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    mean = np.asarray(ops["meanImg"], dtype=np.float32)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}_mean.tif"
    imwrite(str(out), _to_uint16(mean), compression=None)
    guide_src = MC_ROOT / name / "ChanB" / "independent_meanImg.png"
    if guide_src.exists():
        gdir = OUT / "guides"
        gdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(guide_src, gdir / f"{name}_independent_meanImg.png")
        roi = MC_ROOT / name / "ChanB" / "roi_guide_independent_vs_shareA.png"
        if roi.exists():
            shutil.copy2(roi, gdir / f"{name}_roi_guide_independent_vs_shareA.png")
    print(f"  wrote {out}  shape={mean.shape}")
    return out


def prepare() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.txt").write_text(README, encoding="utf-8")
    for name in TRAIN:
        export_fov(name, OUT / "train")
    for name in HOLDOUT:
        export_fov(name, OUT / "holdout")
    print(f"\nLabel folder: {OUT}")
    return OUT / "train" / "LED_x15_Level3b_mean.tif"


def launch_gui(image: Path) -> None:
    """Open Cellpose GUI with *image* loaded.

    Cellpose 3.0.11 sets ``load_3D`` *after* the optional image load, so
    ``gui.run(image=...)`` crashes. Load after init instead.
    """
    print(f"Opening Cellpose GUI: {image}")
    from cellpose.gui import gui as cp_gui
    from cellpose.gui import io as cp_io

    orig_init = cp_gui.MainW.__init__

    def _init(self, image=None, logger=None):
        pending = image
        orig_init(self, image=None, logger=logger)
        self.load_3D = False
        if pending is not None:
            self.filename = pending
            cp_io._load_image(self, self.filename)

    cp_gui.MainW.__init__ = _init
    cp_gui.run(image=str(image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open Cellpose on Level3b mean")
    args = parser.parse_args()
    level3b = prepare()
    if args.gui:
        launch_gui(level3b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
