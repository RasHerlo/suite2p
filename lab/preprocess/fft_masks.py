"""FFT notch helpers for the *legacy* FFT_mask_conversion / master script.

Not used by fringe_robust_register. Defringing belongs upstream
(derippling_PMT_noise), not in the suite2p motion-correction path.
"""

import numpy as np


def mask_from_preset(height, width, preset):
    """Return a boolean (H, W) mask: True = keep, False = notch out."""
    kind = preset["type"]
    coords = preset["coords"]
    if kind == "circle":
        return _circular_mask(height, width, coords)
    if kind == "rect":
        return _rectangular_mask(height, width, coords)
    raise ValueError(f"Unknown FFT mask type {kind!r}")


def apply_fft_notch(frames, preset, batch_size=32):
    """Notch *frames* (n, H, W) in Fourier space. Returns float32."""
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    n, h, w = frames.shape
    keep = mask_from_preset(h, w, preset)
    out = np.empty((n, h, w), dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = frames[i:i + batch_size].astype(np.float32, copy=False)
        spec = np.fft.fftshift(np.fft.fft2(chunk), axes=(-2, -1))
        spec[:, ~keep] = 0
        rec = np.fft.ifft2(np.fft.ifftshift(spec, axes=(-2, -1)))
        out[i:i + batch_size] = np.abs(rec)
    return out


def _rectangular_mask(h, w, mask_coords_list):
    mask = np.ones((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    for x0, y0, dx, dy in mask_coords_list:
        x_min = max(0, int(cx + x0 - dx))
        x_max = min(w, int(cx + x0 + dx))
        y_min = max(0, int(cy + y0 - dy))
        y_max = min(h, int(cy + y0 + dy))
        mask[y_min:y_max, x_min:x_max] = False
    return mask


def _circular_mask(h, w, mask_coords_list):
    mask = np.ones((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_squared = (x - cx) ** 2 + (y - cy) ** 2
    for inner_radius, outer_radius in mask_coords_list:
        ring = (dist_squared < inner_radius ** 2) | (dist_squared > outer_radius ** 2)
        mask &= ring
    return mask
