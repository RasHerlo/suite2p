"""FFT scores that separate cell sharpness from PMT-family re-freezing.

Default QC for motion correction (and, downstream, SUPPORT denoising):

- **Fringe score:** 2D-FFT power inside the defringe signature mask.
  Up after a step = that PMT family was lined up or hallucinated.
- **Cell score:** mid-band power *outside* that mask.
  Up after a step = tissue / soma-scale structure got sharper.

Pass rule (same movie, before vs after): cell power up, fringe power
flat or down. Do **not** notch these frequencies in suite2p; measure only.

Masks are rebuilt from ``signature.json`` written by
``derippling_PMT_noise`` (families + tracking-block ``q`` drift + fx ranges).
That is the same geometry ``family_mask`` uses in the defringe stress tests,
adapted to the JSON that ships with a run (no ``x_weight`` vector).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Tight scoring mask (matches derippling stress_test_v2 tight path).
Y_PAD = 2
DC_EXCLUDE_R = 8
# Mid-band annulus in FFT bins from DC. On 512 px: periods ~64 px (r=8)
# down to ~11 px (r=48). Soma-scale sits here; ChanA's family (~37 px) is
# excluded by subtracting the fringe mask, not by this radius cut.
CELL_R_LO = 8
CELL_R_HI = 48

SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
DEFAULT_SIGNATURE_ROOT = SANDBOX / "defringe_runs" / "v21_full_seeded500"


def load_signature(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_default_signature(letter, start: Path | None = None):
    """ChanA/B signature.json next to a sandbox run, else the v2.1 pack."""
    letter = letter.upper().replace("CHAN", "")
    name = f"Chan{letter}"
    rel = Path("defringe_runs") / "v21_full_seeded500" / name / "diagnostics" / "signature.json"
    if start is not None:
        for p in [Path(start), *Path(start).resolve().parents]:
            cand = p / rel
            if cand.exists():
                return cand
    packed = DEFAULT_SIGNATURE_ROOT / name / "diagnostics" / "signature.json"
    return packed if packed.exists() else None


def _fx_select(width, fam, cx):
    fx = np.arange(width) - cx
    x_sel = np.zeros(width, dtype=bool)
    ranges = fam.get("fx_ranges_weight_gt_0.20") or fam.get("fx_ranges")
    if ranges:
        for lo, hi in ranges:
            x_sel |= (fx >= lo) & (fx <= hi)
        return x_sel
    xw = fam.get("x_weight")
    if xw is not None:
        xw = np.asarray(xw, dtype=float)
        if xw.size == width:
            return xw > 0.20
    return x_sel


def _family_qs(fam, tracking_blocks, family_idx):
    qs = {int(round(float(fam["q"])))}
    hi = fam.get("hi")
    if hi is not None:
        qs.add(int(round(float(hi))))
    for blk in tracking_blocks or []:
        if blk.get("q") is None:
            continue
        if blk.get("family", 0) not in (family_idx, None):
            continue
        qs.add(int(round(float(blk["q"]))))
    return sorted(qs)


def fringe_mask_from_signature(shape, sig, y_pad=Y_PAD, dc_r=DC_EXCLUDE_R):
    """Boolean FFT-centered mask covering the PMT family (and tracked q)."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = h // 2, w // 2
    mask = np.zeros((h, w), dtype=bool)
    blocks = sig.get("tracking_blocks") or []
    for i, fam in enumerate(sig.get("families") or []):
        x_sel = _fx_select(w, fam, cx)
        if not x_sel.any():
            continue
        for q in _family_qs(fam, blocks, i):
            for sgn in (-1, +1):
                yc = cy + sgn * q
                for yp in range(yc - y_pad, yc + y_pad + 1):
                    if 0 <= yp < h:
                        mask[yp, x_sel] = True
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 < dc_r**2] = False
    return mask


def cell_mask_from_fringe(shape, fringe, r_lo=CELL_R_LO, r_hi=CELL_R_HI):
    """Isotropic mid-band minus the fringe family (and DC)."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    cell = (r2 >= r_lo**2) & (r2 <= r_hi**2) & (~np.asarray(fringe, dtype=bool))
    cell[cy, cx] = False
    return cell


def _power_spectrum(img):
    img = np.squeeze(np.asarray(img, dtype=np.float64))
    if img.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {img.shape}")
    img = img - img.mean()
    spec = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(spec) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    power[cy, cx] = 0.0
    return power


def vertical_ridge_energy(img, ky_cut=0.05):
    """Legacy |ky| half-plane fraction. Too wide: scores somata and fringes."""
    power = _power_spectrum(img)
    ly = power.shape[0]
    cy = ly // 2
    ky = (np.arange(ly) - cy) / float(ly)
    ridge = power[np.abs(ky) > ky_cut, :].sum()
    tot = power.sum()
    return float(ridge / (tot + 1e-12))


def band_scores(img, fringe_mask, cell_mask):
    power = _power_spectrum(img)
    tot = float(power.sum())
    fringe_p = float(power[fringe_mask].sum())
    cell_p = float(power[cell_mask].sum())
    return {
        "fringe_power": fringe_p,
        "cell_power": cell_p,
        "total_power": tot,
        "fringe_frac": fringe_p / (tot + 1e-12),
        "cell_frac": cell_p / (tot + 1e-12),
    }


def _ratio(a, b):
    if b is None or b == 0 or not np.isfinite(b):
        return float("nan")
    return float(a / b)


def score_pair(unreg, reg, signature, signature_path=None):
    """Compare unregistered vs registered (or pre vs post) means."""
    sig = signature if isinstance(signature, dict) else load_signature(signature)
    shape = np.squeeze(np.asarray(reg)).shape
    fringe = fringe_mask_from_signature(shape, sig)
    cell = cell_mask_from_fringe(shape, fringe)
    pre = band_scores(unreg, fringe, cell)
    post = band_scores(reg, fringe, cell)
    out = {
        "signature_path": None if signature_path is None else str(signature_path),
        "fringe_n_bins": int(fringe.sum()),
        "cell_n_bins": int(cell.sum()),
        "unreg": pre,
        "reg": post,
        "fringe_power_ratio": _ratio(post["fringe_power"], pre["fringe_power"]),
        "cell_power_ratio": _ratio(post["cell_power"], pre["cell_power"]),
        "fringe_frac_ratio": _ratio(post["fringe_frac"], pre["fringe_frac"]),
        "cell_frac_ratio": _ratio(post["cell_frac"], pre["cell_frac"]),
        "ridge_unreg": vertical_ridge_energy(unreg),
        "ridge_reg": vertical_ridge_energy(reg),
    }
    cell_up = out["cell_power_ratio"] > 1.02
    fringe_ok = out["fringe_power_ratio"] <= 1.05
    if cell_up and fringe_ok:
        out["verdict"] = "cell_up_fringe_ok"
    elif cell_up and not fringe_ok:
        out["verdict"] = "both_up"
    elif (not cell_up) and fringe_ok:
        out["verdict"] = "no_sharpen_fringe_ok"
    else:
        out["verdict"] = "fringe_up_cell_flat"
    return out


def flatten_channel_scores(letter, pair):
    p = pair["unreg"]
    r = pair["reg"]
    L = letter.upper()
    return {
        f"fringe_power_unreg_{L}": p["fringe_power"],
        f"fringe_power_reg_{L}": r["fringe_power"],
        f"fringe_power_ratio_{L}": pair["fringe_power_ratio"],
        f"fringe_frac_unreg_{L}": p["fringe_frac"],
        f"fringe_frac_reg_{L}": r["fringe_frac"],
        f"cell_power_unreg_{L}": p["cell_power"],
        f"cell_power_reg_{L}": r["cell_power"],
        f"cell_power_ratio_{L}": pair["cell_power_ratio"],
        f"cell_frac_unreg_{L}": p["cell_frac"],
        f"cell_frac_reg_{L}": r["cell_frac"],
        f"ridge_unreg_{L}": pair["ridge_unreg"],
        f"ridge_reg_{L}": pair["ridge_reg"],
        f"verdict_{L}": pair["verdict"],
        f"fringe_n_bins_{L}": pair["fringe_n_bins"],
        f"cell_n_bins_{L}": pair["cell_n_bins"],
        f"signature_{L}": pair.get("signature_path"),
    }
