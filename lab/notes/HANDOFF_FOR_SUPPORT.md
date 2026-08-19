# Handoff: suite2p repo → SUPPORT repo

**Audience:** an agent or person working in
[SUPPORT](https://github.com/RasHerlo/SUPPORT)
(denoise defringed stacks; do not re-implement suite2p MC here).

**This file is the outbound note.** Authoritative copy lives in the suite2p
clone: `lab/notes/HANDOFF_FOR_SUPPORT.md`. Sandbox mirror:
`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\support_runs\_handoff\INCOMING_FROM_SUITE2P.md`

Do not extract paper traces. Do not turn on `1Preg`. Do not FFT-notch in
suite2p. Defringe stays in [derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise).

Last updated: 2026-08-19 (late).

---

## Why you need these scores

SUPPORT on `defringed_v21` can sharpen tissue **or** make residual PMT
fringes / tile grids more coherent (already seen: ChanA ridge amp 1.59× and
a box/tile grid on `fullstack_v21_model10`). Eye QC on the mean is not
enough: a prettier mean can be frozen stripes.

Suite2p now scores **two bands** on the 2D mean (same images you would
compare: input mean vs output mean). Use the same scores for
**defringed vs denoised**. Pass: cell-band up or flat, PMT-family power
**not** up.

---

## Metrics

2D-FFT of a **mean image**, DC bin zeroed.

| Score | What | Up means |
|---|---|---|
| **Fringe power** | `sum(|FFT|²)` inside the defringe **signature mask** | that PMT family was lined up or hallucinated |
| **Cell power** | `sum(|FFT|²)` in a mid-band annulus **outside** that mask (FFT radius 8–48 bins on 512 px ≈ 64–11 px periods) | soma / tissue sharper |

**Pass (same FOV, same crop):** `cell_power_ratio = post/pre > 1` (or not much
down) **and** `fringe_power_ratio ≲ 1`.

**Do not use** fringe *fraction of total power* as the pass. After MC, ChanA
family **fraction** fell (0.113 → 0.098) while **absolute** family power
rose 1.9×, because cells rose too. Report **power ratios**.

Legacy `|ky|>0.05` half-plane fraction is **too wide**. ChanA’s v2.1 family
sits at `|ky|≈0.027` (below the cut); the old score mixed somata with
fringes.

**Not in this mask:** SUPPORT box/tile grids. Those are a different lattice.
Keep a visual (or a separate grid score). A PMT-family pass can still fail
on tiles.

Do **not** notch these frequencies inside SUPPORT or suite2p. Measure only.

---

## Signature files (rebuild the mask from these)

Use the signature of the stack you denoised, not a guess at `q`.

For current `inputs/defringed_v21/` (v2.1 pack):

```
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\defringe_runs\v21_full_seeded500\ChanA\diagnostics\signature.json
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\defringe_runs\v21_full_seeded500\ChanB\diagnostics\signature.json
```

If you switch to v2.2 stacks, use `defringe_runs/v22_full_seeded500/...`
instead.

JSON fields used: `families[].q`, `families[].hi`,
`families[].fx_ranges_weight_gt_0.20`, and all `tracking_blocks[].q` for
that family (q drifts; union them). Geometry matches defringe
`family_mask` (tight: `y_pad=2`, DC disk radius 8 excluded).

Seed (v2.1): ChanA `q≈14` (`fx` ±10–38, ~37 px); ChanB `q≈60` (`fx` ±10–41,
~8.5 px).

---

## Code to copy or import

Implementation (suite2p lab, no MouseLand patches):

`C:\Users\rasmu\Projects\Repos\suite2p\lab\pipeline\mc_fft_metrics.py`

Interpreter if you import from that repo:
`C:\Users\rasmu\anaconda3\envs\suite2p\python.exe`

Minimal use on two means (e.g. defringed mean vs SUPPORT mean):

```python
from pathlib import Path
import numpy as np
from tifffile import imread
import sys
sys.path.insert(0, r"C:\Users\rasmu\Projects\Repos\suite2p")
from lab.pipeline.mc_fft_metrics import load_signature, score_pair

def mean_of(path):
    x = np.squeeze(np.asarray(imread(str(path), is_ome=False)))
    return x.mean(axis=0) if x.ndim == 3 else x

sig = load_signature(Path(r".../ChanA/diagnostics/signature.json"))
pair = score_pair(mean_of("defringed.tif"), mean_of("denoised.tif"), sig)
print(pair["cell_power_ratio"], pair["fringe_power_ratio"], pair["verdict"])
```

`verdict` strings: `cell_up_fringe_ok` (want this), `both_up` (cells sharper
but family also up — fail for promote), `no_sharpen_fringe_ok`,
`fringe_up_cell_flat`.

If you do not want a suite2p import, copy `mc_fft_metrics.py` into SUPPORT
and keep the mask builder in sync with `signature.json`.

---

## What to compare

| pre | post |
|---|---|
| temporal mean of the **defringed** stack you fed SUPPORT | temporal mean of the **denoised** stack |

Same plane, same crop. If SUPPORT is 5340 frames vs 5400 input, say so and
mean only the overlapping stretch (or the mirrored full-length output).
Do not mix ChanA signature with a ChanB image.

Optional later (suite2p’s job): the same scores on unregistered vs
registered **after** your denoise. Tonight’s MC on raw and on v21 (no
SUPPORT) is already `both_up` — denoise must not make the family worse
before that step.

---

## Tonight’s MC numbers (context, not your job to re-run)

Cell-ops register on Level3b copy, honest unreg mean, v2.1 signatures.
Ratios are **reg / unreg** power.

| run | cell A/B | fringe A/B |
|---|---|---|
| `mc_runs/raw_cell` | 1.95 / 1.34 | 1.88 / 10.5 |
| `mc_runs/v21_cell` | 1.95 / 1.33 | 1.90 / 13.0 |
| `mc_runs/v21_cell_shareA` | 1.95 / 1.35 | 1.90 / 3.91 |

Registered means look crispier (cell band). Independent ChanB strongly
re-freezes its family. That is why a denoise that **also** raises family
power is a hard fail.

Figures: `mc_runs/<run>/compare_AB.png` + `.json`.

---

## Do not

- Do not treat a higher `cmax` or a crisp mean as success.
- Do not promote `support_runs/fullstack_v21_model10/` (already: ChanA
  fringe amp + tile grid).
- Do not share MC shifts across 5400 vs 5340 lengths.
- Do not overwrite `inputs/raw`, `inputs/defringed`, `inputs/defringed_v21`.
