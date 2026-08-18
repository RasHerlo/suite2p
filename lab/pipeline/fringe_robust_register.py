#!/usr/bin/env python3
"""Cell-oriented motion correction on stacks delivered to this repo.

Defringing is *not* done here. Incoming TIFFs may already have been defringed
upstream (derippling_PMT_noise). This script only estimates and applies
suite2p shifts, with ops chosen so phase correlation tracks cells rather
than residual high-frequency texture.

Does not enable 1Preg. Writes suite2p_cellreg/ so legacy suite2p files/
is left alone.

Usage
-----
    python lab/pipeline/fringe_robust_register.py <parent_folder>
    python lab/pipeline/fringe_robust_register.py path/to/stack.tif --channel A
    python lab/pipeline/fringe_robust_register.py <parent> --no-share-shifts

See lab/notes/motion_correction.md.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imwrite

from suite2p import default_ops
from suite2p.registration import register

from lab.configs.defaults import REGISTRATION, REGISTRATION_LEGACY


def channel_letter_from_path(path):
    """ChanA / SUPPORT_ChanA / ChanA_defringe / ... → 'A' or 'B', else None."""
    parts = Path(path).parts
    for token in reversed(parts):
        if "ChanA" in token and "ChanB" not in token:
            return "A"
        if "ChanB" in token:
            return "B"
    return None


def to_int16_s2p(mov):
    """Match process_registration.py dtype conversion."""
    mov = np.asarray(mov)
    if mov.ndim == 2:
        mov = mov[np.newaxis, ...]
    if mov.dtype.type == np.uint16:
        return (mov // 2).astype(np.int16)
    if mov.dtype.type == np.int32:
        return (mov // 2).astype(np.int16)
    if mov.dtype.type == np.int16:
        return mov
    m = mov.astype(np.float32)
    lo, hi = np.percentile(m, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((m - lo) / (hi - lo), 0, 1) * 32767.0
    return scaled.astype(np.int16)


def damp_weak_shifts(yoff, xoff, cmax, frac=0.4):
    """Interpolate shifts on frames whose phase-corr peak is weak vs the median."""
    yoff = np.asarray(yoff, dtype=np.float64).copy()
    xoff = np.asarray(xoff, dtype=np.float64).copy()
    cmax = np.asarray(cmax, dtype=np.float64)
    med = np.median(cmax) if cmax.size else 0.0
    good = cmax >= (frac * med) if med > 0 else np.ones(cmax.shape, dtype=bool)
    n_good = int(np.sum(good))
    if n_good < 3 or n_good == len(good):
        return yoff, xoff, good
    t = np.arange(len(yoff))
    yoff[~good] = np.interp(t[~good], t[good], yoff[good])
    xoff[~good] = np.interp(t[~good], t[good], xoff[good])
    return yoff, xoff, good


def make_alignment_movie(mov, channel_letter, cfg):
    """Optional extra blur for shift *estimation* only (not a defringe product).

    The delivered movie is what we keep. A spatial low-pass here is only to
    down-weight residual high-frequency texture in phase correlation.
    """
    filt = cfg.get("align_filter", "none")
    if filt in (None, "none", ""):
        print("    alignment filter: none (delivered stack)")
        return to_int16_s2p(mov)
    if filt != "lowpass":
        raise ValueError(
            f"Unknown align_filter {filt!r}. Use 'none' or 'lowpass'. "
            "FFT notch / defringe is not performed in this repo."
        )
    sigma = float(cfg.get("lowpass_sigma", 4.0))
    print(f"    alignment low-pass sigma={sigma} (phasecorr weighting, not defringe)")
    blurred = gaussian_filter(mov.astype(np.float32), sigma=(0.0, sigma, sigma))
    return to_int16_s2p(blurred)


def build_ops(output_dir, cfg, nframes, ly, lx):
    ops = default_ops()
    ops["nplanes"] = 1
    ops["nchannels"] = 1
    ops["do_registration"] = True
    ops["nonrigid"] = bool(cfg.get("nonrigid", False))
    ops["1Preg"] = bool(cfg.get("one_p_reg", False))
    ops["smooth_sigma"] = float(cfg.get("smooth_sigma", 3.0))
    ops["maxregshift"] = float(cfg.get("maxregshift", 0.1))
    ops["maxregshiftNR"] = int(cfg.get("maxregshiftNR", 3))
    ops["reg_tif"] = False
    ops["batch_size"] = min(500, nframes)
    ops["nimg_init"] = min(300, nframes)
    ops["save_path"] = str(output_dir)
    ops["save_path0"] = str(output_dir)
    ops["nframes"] = nframes
    ops["Ly"] = ly
    ops["Lx"] = lx
    return ops


def estimate_and_apply(mov, channel_letter, output_dir, cfg, offsets=None):
    """Register *mov*. If *offsets* is given, skip estimation and only apply."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mov = to_int16_s2p(mov)
    nframes, ly, lx = mov.shape
    ops = build_ops(output_dir, cfg, nframes, ly, lx)

    if offsets is None:
        align = make_alignment_movie(mov, channel_letter, cfg)
        print("    estimating shifts")
        _ref, _rmin, _rmax, _mean_a, rigid_off, _nr, _z = (
            register.compute_reference_and_register_frames(align, ops=ops)
        )
        yoff, xoff, cmax = rigid_off
        yoff1 = xoff1 = None
        if ops["nonrigid"] and _nr:
            yoff1, xoff1, _cmax1 = _nr
        if ops["nonrigid"] or float(cfg.get("corrxy_frac", 0.4)) <= 0:
            good = np.ones(np.shape(cmax), dtype=bool)
            n_damped = 0
        else:
            yoff, xoff, good = damp_weak_shifts(
                yoff, xoff, cmax, frac=float(cfg.get("corrxy_frac", 0.4))
            )
            n_damped = int(np.size(good) - np.sum(good))
        print(f"    cmax median={np.median(cmax):.4f}  interpolated {n_damped} weak frames")
    else:
        yoff = np.asarray(offsets["yoff"])
        xoff = np.asarray(offsets["xoff"])
        cmax = np.asarray(offsets["cmax"])
        good = np.asarray(offsets["good"])
        yoff1 = offsets.get("yoff1")
        xoff1 = offsets.get("xoff1")
        print("    applying shared shifts (no re-estimation)")

    if len(yoff) != nframes:
        raise ValueError(
            f"shift length {len(yoff)} != nframes {nframes} "
            "(shared ChanA/B stacks must be the same length)"
        )

    kept = deepcopy(ops)
    registered = mov.astype(np.float32, copy=True)
    mean_img = register.shift_frames_and_write(
        registered,
        f_alt_out=registered,
        yoff=np.round(yoff).astype(np.int32),
        xoff=np.round(xoff).astype(np.int32),
        yoff1=yoff1,
        xoff1=xoff1,
        ops=kept,
    )

    if cfg.get("write_registered_tif", True):
        out_tif = output_dir / "combined_registered.tif"
        imwrite(str(out_tif), registered.astype(np.int16))
        print(f"    wrote {out_tif}")
    else:
        print("    skip combined_registered.tif (write_registered_tif=False)")

    ops_save = deepcopy(ops)
    ops_save["yoff"] = yoff
    ops_save["xoff"] = xoff
    ops_save["corrXY"] = cmax
    ops_save["meanImg"] = mean_img
    ops_save["align_channel"] = channel_letter
    ops_save["align_filter"] = cfg.get("align_filter")
    orig_mean = np.mean(mov, axis=0)
    np.save(output_dir / "mean_unregistered.npy", orig_mean.astype(np.float32))
    np.save(output_dir / "ops.npy", ops_save)
    savez = dict(yoff=yoff, xoff=xoff, cmax=cmax, good=good)
    if yoff1 is not None:
        savez["yoff1"] = yoff1
        savez["xoff1"] = xoff1
    np.savez(output_dir / "offsets.npz", **savez)
    _save_diagnostics(output_dir, mov, registered, yoff, xoff, cmax, good, mean_img)
    return {
        "yoff": yoff,
        "xoff": xoff,
        "cmax": cmax,
        "good": good,
        "yoff1": yoff1,
        "xoff1": xoff1,
        "mean_img": mean_img,
    }


def _save_diagnostics(output_dir, original, registered, yoff, xoff, cmax, good, mean_img):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    t = np.arange(len(yoff))
    axes[0, 0].plot(t, yoff, label="yoff")
    axes[0, 0].plot(t, xoff, label="xoff")
    axes[0, 0].set_title("rigid shifts (px)")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].plot(t, cmax, color="0.3")
    if not np.all(good):
        axes[0, 1].plot(t[~good], cmax[~good], "r.", label="interpolated")
        axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_title(f"phase-corr peak  median={np.median(cmax):.4f}")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        0.5,
        "Delivered stack is registered as-is (defringe is upstream).\n"
        "smooth_sigma / optional lowpass only change what phasecorr sees.\n"
        "1Preg stays off. Residual fringes should not get sharper.",
        transform=axes[0, 2].transAxes,
        va="center",
    )

    orig_mean = original.mean(axis=0)
    for ax, img, title in (
        (axes[1, 0], orig_mean, "mean original"),
        (axes[1, 1], mean_img, "mean registered"),
        (axes[1, 2], mean_img.astype(np.float32) - orig_mean.astype(np.float32), "registered − original"),
    ):
        lo, hi = np.percentile(img, (1, 99))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostics_register.png", dpi=120)
    plt.close(fig)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_img, cmap="coolwarm")
    plt.axis("off")
    plt.savefig(output_dir / "MeanImg.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def find_input_tiff(folder, cfg, letter=None):
    folder = Path(folder)
    letter = letter or channel_letter_from_path(folder)
    names = []
    if letter == "A":
        names.extend([
            "ChanA_stk_defringed_v21.tif",
            "ChanA_stk.tif",
            "ChanA_stk_defringed.tif",
            "denoised_cut.tif",
        ])
    elif letter == "B":
        names.extend([
            "ChanB_stk_defringed_v21.tif",
            "ChanB_stk.tif",
            "ChanB_stk_defringed.tif",
            "denoised_cut.tif",
        ])
    else:
        names.extend(cfg.get("input_tiff_names", ()))
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def process_tiff(tif_path, output_dir=None, channel_letter=None, cfg=None, offsets=None):
    cfg = cfg or deepcopy(REGISTRATION)
    tif_path = Path(tif_path)
    channel_letter = channel_letter or channel_letter_from_path(tif_path) or cfg.get("align_channel", "A")
    output_dir = Path(output_dir or (tif_path.parent / cfg.get("output_folder", "suite2p_cellreg")))
    print(f"\n  {tif_path}  channel={channel_letter}")
    mov = imread(str(tif_path), is_ome=False)
    return estimate_and_apply(mov, channel_letter, output_dir, cfg, offsets=offsets)


def _support_folders(root):
    root = Path(root)
    found = {"A": [], "B": []}
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        letter = channel_letter_from_path(folder)
        if letter in found and find_input_tiff(folder, REGISTRATION, letter=letter):
            found[letter].append(folder)
    return found


def _pair_key(folder):
    """DATA parent so ChanA/B from the same run are paired."""
    p = Path(folder)
    if p.name.startswith("SUPPORT_"):
        return str(p.parent)
    return str(p.parent)


def process_tree(root, cfg, share_shifts=True, overwrite=False, output_root=None):
    found = _support_folders(root)
    groups = {}
    for letter, folders in found.items():
        for folder in folders:
            key = _pair_key(folder)
            pair = groups.setdefault(key, {})
            previous = pair.get(letter)
            prefer_support = Path(folder).name.startswith("SUPPORT_")
            prev_support = previous is not None and Path(previous).name.startswith("SUPPORT_")
            if previous is None or (prefer_support and not prev_support):
                pair[letter] = folder

    if not groups:
        # maybe a single tiff was passed
        root = Path(root)
        if root.is_file() and root.suffix.lower() in {".tif", ".tiff"}:
            process_tiff(root, cfg=cfg)
            return
        print(f"No ChanA/B stacks found under {root}")
        return

    align_letter = cfg.get("align_channel", "A")
    n = 0
    for key, pair in sorted(groups.items()):
        print(f"\n=== {key} ===")
        offsets = None
        order = []
        if share_shifts and align_letter in pair:
            order = [align_letter] + [L for L in pair if L != align_letter]
        else:
            order = sorted(pair.keys())

        for letter in order:
            folder = pair[letter]
            tif = find_input_tiff(folder, cfg, letter=letter)
            if output_root is not None:
                out = Path(output_root) / f"Chan{letter}"
            else:
                out = folder / cfg.get("output_folder", "suite2p_cellreg")
            already = (out / "offsets.npz").exists() or (
                out.exists() and (out / "combined_registered.tif").exists()
            )
            if already and not overwrite:
                print(f"  skip existing {out}")
                if share_shifts and letter == align_letter and offsets is None:
                    z = np.load(out / "offsets.npz")
                    offsets = {k: z[k] for k in ("yoff", "xoff", "cmax", "good")}
                continue
            apply = offsets if (share_shifts and letter != align_letter) else None
            result = process_tiff(tif, output_dir=out, channel_letter=letter, cfg=cfg, offsets=apply)
            if share_shifts and letter == align_letter:
                offsets = result
            n += 1
    print(f"\nFinished {n} stack(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Cell-oriented suite2p motion correction on a delivered stack."
    )
    parser.add_argument("input_path", help="Parent folder (walks ChanA/B) or a .tif")
    parser.add_argument("--channel", choices=["A", "B"], default=None,
                        help="PMT letter when registering a single tiff")
    parser.add_argument("--profile", choices=["cell", "legacy"], default="cell",
                        help="cell = REGISTRATION; legacy = stock-like nonrigid")
    parser.add_argument("--no-share-shifts", action="store_true",
                        help="Estimate each channel independently")
    parser.add_argument("--output-root", default=None,
                        help="Write ChanA/ChanB under this folder instead of beside the tiff")
    parser.add_argument("--save-stack", action="store_true",
                        help="Also write combined_registered.tif (large)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if offsets.npz already exists")
    args = parser.parse_args()
    cfg = deepcopy(REGISTRATION_LEGACY if args.profile == "legacy" else REGISTRATION)
    if args.save_stack:
        cfg["write_registered_tif"] = True
    if args.no_share_shifts:
        cfg["share_shifts_across_channels"] = False
    path = Path(args.input_path)
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return 1
    output_root = Path(args.output_root) if args.output_root else None
    if path.is_file():
        out = None
        if output_root is not None:
            letter = args.channel or channel_letter_from_path(path) or "A"
            out = output_root / f"Chan{letter}"
        process_tiff(path, output_dir=out, channel_letter=args.channel, cfg=cfg)
        return 0
    process_tree(
        path,
        cfg,
        share_shifts=cfg["share_shifts_across_channels"],
        overwrite=args.overwrite,
        output_root=output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
