#!/usr/bin/env python3
"""
basic_suite2p_walk.py
=====================
Walk a root directory, find all experiment DATA folders, and run the full
suite2p pipeline (registration → ROI detection → signal extraction →
spike deconvolution → ROI selection) on all four channel stacks:

    DATA/
      ChanA/          → ChanA_stk.tif
      ChanB/          → ChanB_stk.tif
      SUPPORT_ChanA/  → denoised_cut.tif   (denoised.tif deleted if found)
      SUPPORT_ChanB/  → denoised_cut.tif   (denoised.tif deleted if found)

Suite2p output is written to a 'suite2p' subfolder inside the same folder
as the input .tif.  If a 'suite2p' folder already exists the dataset is
skipped with a warning.

Usage
-----
    python -m lab.pipeline.basic_suite2p_walk <root_dir>
    python lab/pipeline/basic_suite2p_walk.py <root_dir>
    python lab/pipeline/basic_suite2p_walk.py <root_dir> --overwrite

Settings
--------
    From lab/configs/defaults.py (S2P_OPS + ROI_SELECTION).
"""

import os
import sys
import logging
import argparse
import warnings
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Show suite2p's INFO-level progress messages in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless – no display needed
import matplotlib.pyplot as plt

from suite2p import default_ops
from suite2p.run_s2p import run_s2p
from lab.configs.defaults import CHANNEL_FILES, ROI_SELECTION, apply_s2p_ops
from lab.detection.roi_selection_new import ROISelector


def build_ops(tif_path, output_dir):
    """Return a fully-configured ops dict for the given input file."""
    return apply_s2p_ops(default_ops(), tif_path=tif_path, output_dir=output_dir)


# ─── ROI selection (same thresholds as data_processing_master.py) ─────────────

def run_roi_selection(plane0_dir):
    """
    Apply ellipticity + connected-component filtering and overwrite iscell.npy.
    Also saves roi_selection.png in the parent folder.
    """
    selector = ROISelector(plane0_dir)
    n_before = int(np.sum(selector.iscell[:, 0]))

    new_iscell = selector.apply_selection_function(
        'select_by_roi_ellipticity_and_components',
        ellipticity_threshold=ROI_SELECTION['ellipticity_threshold'],
        components_threshold=ROI_SELECTION['components_threshold'],
        show_plot=False,
    )
    np.save(Path(plane0_dir) / 'iscell.npy', new_iscell)
    selector.iscell = new_iscell

    n_after = int(np.sum(new_iscell[:, 0]))
    print(f'    ROI selection: {n_before} → {n_after} cells kept')

    # save visualisation alongside the suite2p folder
    _save_roi_vis(selector, Path(plane0_dir).parent.parent)


def _save_roi_vis(selector, out_dir):
    mean_img = selector.ops['meanImg']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(mean_img, cmap='gray')
    for i, roi in enumerate(selector.stat):
        if selector.iscell[i, 0]:
            ax1.plot(roi['xpix'], roi['ypix'], '.', markersize=1, alpha=0.5)
    n_sel = int(np.sum(selector.iscell[:, 0]))
    ax1.set_title(f'Selected ROIs ({n_sel})')
    ax1.axis('off')

    ax2.imshow(mean_img, cmap='gray')
    for i, roi in enumerate(selector.stat):
        if not selector.iscell[i, 0]:
            ax2.plot(roi['xpix'], roi['ypix'], '.', markersize=1, alpha=0.5)
    n_rej = len(selector.stat) - n_sel
    ax2.set_title(f'Rejected ROIs ({n_rej})')
    ax2.axis('off')

    plt.tight_layout()
    fig_path = out_dir / 'roi_selection.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    ROI visualisation → {fig_path}')


# ─── Per-stack processing ──────────────────────────────────────────────────────

def process_stack(tif_path, overwrite=False):
    """
    Run suite2p + ROI selection on a single .tif stack.

    Output is written to  <tif_dir>/suite2p/plane0/.
    Skips (with warning) if the suite2p folder already exists and
    --overwrite was not requested.

    Returns True on success, False on skip/error.
    """
    tif_path   = Path(tif_path)
    tif_dir    = tif_path.parent
    s2p_dir    = tif_dir / 'suite2p'
    plane0_dir = s2p_dir / 'plane0'

    # ── skip guard ────────────────────────────────────────────────────────────
    if s2p_dir.exists() and not overwrite:
        warnings.warn(
            f'suite2p folder already exists, skipping: {s2p_dir}',
            stacklevel=2
        )
        return False

    print(f'\n  Processing: {tif_path}')
    t0 = datetime.datetime.now()

    # ── delete denoised.tif if present (only relevant for SUPPORT folders) ────
    denoised = tif_dir / 'denoised.tif'
    if denoised.exists():
        print(f'    Deleting {denoised}')
        denoised.unlink()

    # ── run suite2p ───────────────────────────────────────────────────────────
    ops = build_ops(tif_path, tif_dir)
    try:
        run_s2p(ops=ops)
    except Exception as exc:
        print(f'    ERROR during suite2p: {exc}')
        return False

    # ── ROI selection ─────────────────────────────────────────────────────────
    if not plane0_dir.exists():
        print(f'    WARNING: expected plane0 dir not found at {plane0_dir}')
        return False

    try:
        run_roi_selection(plane0_dir)
    except Exception as exc:
        print(f'    WARNING: ROI selection failed: {exc}')

    elapsed = datetime.datetime.now() - t0
    print(f'    Done in {elapsed}')
    return True


# ─── Folder discovery ──────────────────────────────────────────────────────────

_CHANNEL_FILES = CHANNEL_FILES


def find_targets(root):
    """
    Walk *root* and yield (tif_path, channel_label) for every channel stack
    found inside a DATA subfolder of any experiment.

    Expected structure:
        <root>/
          <experiment>/
            DATA/
              ChanA/           → ChanA_stk.tif
              ChanB/           → ChanB_stk.tif
              SUPPORT_ChanA/   → denoised_cut.tif
              SUPPORT_ChanB/   → denoised_cut.tif
    """
    root = Path(root)
    for data_dir in sorted(root.rglob('DATA')):
        if not data_dir.is_dir():
            continue
        for chan_name, tif_name in _CHANNEL_FILES.items():
            tif_path = data_dir / chan_name / tif_name
            if tif_path.exists():
                yield tif_path, chan_name
            else:
                # report so the user knows something was expected but missing
                expected = data_dir / chan_name / tif_name
                print(f'  [not found] {expected}')


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch suite2p walk — processes all channel stacks '
                    'found under DATA folders in the root directory.')
    parser.add_argument('root_dir',
                        help='Root directory containing experiment folders')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-process even if a suite2p folder already exists')
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        print(f'ERROR: root directory does not exist: {root}')
        sys.exit(1)

    targets = list(find_targets(root))
    if not targets:
        print('No target stacks found. Check that your DATA folders exist '
              'and contain the expected channel subfolders.')
        sys.exit(0)

    print(f'\nFound {len(targets)} stack(s) to process under {root}\n')
    for tif_path, chan in targets:
        print(f'  [{chan}]  {tif_path}')

    print('\n' + '=' * 70)

    n_ok = n_skip = n_err = 0
    t_start = datetime.datetime.now()

    for tif_path, chan in targets:
        print(f'\n[{chan}] {tif_path.parent}')
        result = process_stack(tif_path, overwrite=args.overwrite)
        if result is True:
            n_ok += 1
        elif result is False:
            # distinguish skipped (suite2p exists) from error
            s2p = tif_path.parent / 'suite2p'
            if s2p.exists() and not args.overwrite:
                n_skip += 1
            else:
                n_err += 1

    total = datetime.datetime.now() - t_start
    print('\n' + '=' * 70)
    print(f'Finished.  OK: {n_ok}  |  Skipped: {n_skip}  |  Errors: {n_err}')
    print(f'Total time: {total}')


if __name__ == '__main__':
    main()
