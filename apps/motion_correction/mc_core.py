"""Single-channel motion correction using the stock Suite2p registration.

No lab-specific logic (no defringe handling, no two-channel shift sharing).
One TIFF stack in -> registered TIFF stack + CSV of per-frame shifts out.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import tifffile

# torch and numpy/MKL can each ship an OpenMP runtime on Windows; avoid the hard abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Parameters exposed in the GUI (everything else stays at Suite2p defaults).
DEFAULT_PARAMS = {
    "nonrigid": True,
    "block_size": 128,
    "maxregshift": 0.1,
    "smooth_sigma": 1.15,
    "nimg_init": 300,
    "batch_size": 500,
    "do_bidiphase": False,
    "1Preg": False,
}


def _load_registration():
    """Import suite2p.registration.register without importing the full suite2p package.

    `import suite2p` pulls in cellpose, rastermap, nwb, etc. which are not needed
    for registration and would bloat the .exe. We register lightweight stand-in
    packages so only the registration modules are loaded.
    """
    if "suite2p.registration.register" in sys.modules:
        return sys.modules["suite2p.registration.register"]

    root = None
    for cand in (getattr(sys, "_MEIPASS", None), str(REPO_ROOT)):
        if cand and (Path(cand) / "suite2p" / "registration" / "register.py").exists():
            root = Path(cand)
            break
    if root is None:  # fall back to an installed suite2p
        from suite2p.registration import register
        return register

    def _pkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
        return m

    s2p = _pkg("suite2p", root / "suite2p")
    ver = types.ModuleType("suite2p.version")
    ver.version = "bundled"
    sys.modules["suite2p.version"] = ver
    s2p.version = ver.version

    import importlib
    dops = importlib.import_module("suite2p.default_ops")
    s2p.default_ops = dops.default_ops

    io_pkg = _pkg("suite2p.io", root / "suite2p" / "io")
    binary = importlib.import_module("suite2p.io.binary")
    io_pkg.BinaryFile = binary.BinaryFile
    s2p.io = io_pkg

    _pkg("suite2p.registration", root / "suite2p" / "registration")
    return importlib.import_module("suite2p.registration.register")


def _default_ops():
    _load_registration()
    return sys.modules["suite2p.default_ops"].default_ops()


def _to_int16(mov: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert to int16 as Suite2p expects. Returns (array, note)."""
    if mov.dtype == np.int16:
        return mov, ""
    if mov.dtype in (np.uint16, np.int32, np.uint32):
        return (mov // 2).astype(np.int16), "intensities halved to fit int16"
    m = mov.astype(np.float32)
    lo, hi = np.percentile(m, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = (np.clip((m - lo) / (hi - lo), 0, 1) * 32767).astype(np.int16)
    return out, "intensities rescaled to int16"


def motion_correct(tif_path, out_dir=None, params=None, log=print, cancel=lambda: False):
    """Register one TIFF stack. Returns dict of output paths."""
    reg_mod = _load_registration()
    tif_path = Path(tif_path)
    out_dir = Path(out_dir) if out_dir else tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {**DEFAULT_PARAMS, **(params or {})}

    log(f"Reading {tif_path.name} ...")
    mov = tifffile.imread(str(tif_path))
    mov = np.squeeze(mov)
    if mov.ndim != 3:
        raise ValueError(f"Expected a stack (frames, y, x); got shape {mov.shape}")
    n, ly, lx = mov.shape
    log(f"Stack: {n} frames, {ly} x {lx}, dtype {mov.dtype}")
    mov, note = _to_int16(mov)
    if note:
        log(f"Note: {note} (shifts are unaffected).")

    ops = _default_ops()
    ops.update(
        nonrigid=bool(p["nonrigid"]),
        block_size=[int(p["block_size"])] * 2,
        maxregshift=float(p["maxregshift"]),
        smooth_sigma=float(p["smooth_sigma"]),
        nimg_init=int(p["nimg_init"]),
        batch_size=int(p["batch_size"]),
        do_bidiphase=bool(p["do_bidiphase"]),
        **{"1Preg": bool(p["1Preg"])},
    )
    ops["nimg_init"] = min(ops["nimg_init"], n)
    if ops["1Preg"]:
        ops["smooth_sigma"] = max(ops["smooth_sigma"], 3.0)

    tmp = tempfile.mkdtemp(prefix="s2p_mc_")
    try:
        f_raw = np.memmap(os.path.join(tmp, "raw.bin"), dtype="int16", mode="w+",
                          shape=mov.shape)
        f_raw[:] = mov
        f_raw.flush()
        del mov
        f_reg = np.memmap(os.path.join(tmp, "reg.bin"), dtype="int16", mode="w+",
                          shape=f_raw.shape)

        if cancel():
            raise InterruptedError("Cancelled")
        log("Registering (Suite2p rigid%s) ..." % (" + non-rigid" if ops["nonrigid"] else ""))
        out = reg_mod.registration_wrapper(f_reg, f_raw=f_raw, refImg=None,
                                           ops=ops)
        (_ref, _rmin, _rmax, mean_img, rigid, nonrigid, _z, _m2, badframes,
         yrange, xrange) = out
        yoff, xoff, corr = rigid

        stem = tif_path.stem
        reg_path = out_dir / f"{stem}_registered.tif"
        csv_path = out_dir / f"{stem}_shifts.csv"

        log("Writing registered stack ...")
        # ImageJ cannot open BigTIFF, so write a classic ImageJ-compatible TIFF
        # whenever the stack fits under the 4 GB limit.
        big = f_reg.nbytes > 3.9 * 1024**3
        if big:
            log("Warning: stack > 4 GB; writing BigTIFF (open in ImageJ via Bio-Formats).")
        tifffile.imwrite(str(reg_path), f_reg, imagej=not big, bigtiff=big,
                         metadata={"axes": "TYX"} if not big else None)
        tifffile.imwrite(str(out_dir / f"{stem}_meanImg.tif"), mean_img.astype(np.float32))

        log("Writing shifts CSV ...")
        header = ["frame", "yoff", "xoff", "corrXY", "badframe"]
        cols = [np.arange(n), yoff, xoff, corr, np.asarray(badframes).astype(int)]
        if ops["nonrigid"] and len(nonrigid) == 3:
            y1, x1, c1 = (np.asarray(a) for a in nonrigid)
            nb = y1.shape[1]
            header += [f"yoff_block{b}" for b in range(nb)]
            header += [f"xoff_block{b}" for b in range(nb)]
            cols += [y1[:, b] for b in range(nb)] + [x1[:, b] for b in range(nb)]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n):
                w.writerow([int(cols[0][i])] + [f"{float(c[i]):.4f}" if j != 3 else int(c[i])
                                                for j, c in enumerate(cols[1:])])
        log(f"Done. Max |shift|: y {np.abs(yoff).max():.1f}, x {np.abs(xoff).max():.1f} px")
        del f_raw, f_reg
        return {"registered": str(reg_path), "shifts": str(csv_path)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
