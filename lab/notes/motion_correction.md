# Motion correction vs PMT fringes

Last updated: 2026-08-18

## Why this is first

Electronic Moire / PMT ripples live in a **scan/PMT layer**, not on the mouse.
Suite2p phase correlation can lock onto that texture. Then:

- cells are shifted as if they were the fringes (biology is warped)
- the registered mean **lines the ripples up**, so they look worse than `stk_avg`

Seen on `acq-260511-led-level1` (Shinano, LED_x15_Level1):

- `corrXY` ~0.01 (ChanA) / ~0.005 (ChanB) — weak lock
- ChanA vs ChanB **xoff** Pearson ~0.93, **yoff** ~0.07
- PMT ridges sit in Fourier-y → y-shifts are the plausible fringe-lock axis
- `spatial_hp_reg=42` (only if `1Preg`) **keeps** ~10 px structure (fringe band)
  and **removes** large astrocyte clouds — the wrong filter for this noise
- SUPPORT iscell on that FOV is fringe-dominated (ChanA ~76% of FOV)

Sources: `figure_for_cAMP_Neu_paper/catalog/preprocessing/TODO.md`,
`figures/fig01_neighboring_neural_activity/layout/led_x15_level1/README.md`.
Defringe work lives in https://github.com/RasHerlo/derippling_PMT_noise
and is **not** run in this repo. Stacks should arrive already defringed
if that step is wanted. Residual fringes can still bias phasecorr; that
is a motion-correction ops problem, not a reason to notch here.

## What suite2p is doing (do not patch MouseLand for this)

Phase correlation **whitens** Fourier amplitude, so high-frequency periodic
structure (fringes) dominates the peak even when `1Preg` is off.
`smooth_sigma` (~1.15 default) is a Gaussian in that FFT; 1P docs already
recommend 3–5 to ignore fine junk.

Nonrigid (128 px blocks, `maxregshiftNR=5`) can locally warp onto stripes.

`compute_reference_and_register_frames` always **shifts the array it was
given**. There is already `shift_frames_and_write` to apply those offsets
to a second movie — that is the two-stream API we want.

## Strategy (this repo: register + segment only)

Defringe upstream if needed. Here, register the **delivered** movie:

```
delivered stack (maybe already defringed)
        │
        ▼
  suite2p rigid phasecorr
    smooth_sigma=3 (ignore fine residual texture)
    1Preg=False (do not high-pass-keep a ~10 px band)
    nonrigid=False for the first pass
    interpolate frames with weak cmax
        │
        ▼
  registered movie (same intensities, shifted)
```

Optional `align_filter=lowpass`: estimate shifts on a slightly blurred copy
so phasecorr attends to soma-scale structure. That is **not** a defringe
product; the kept movie is still the delivered stack.

Optional: compute the trace on the neuron-like PMT and apply to both
channels (`share_shifts_across_channels`). ChanA/B are PMT paths, not cell
types — set the align channel per experiment.

**Do not use `1Preg` / `spatial_hp_reg` here.** That high-pass keeps residual
fringe-scale texture.

**Do not FFT-notch here.** The legacy ChanA ring / ChanB rects were a
stand-in for defringing inside this repo; they are out of scope.

## What we implemented (2026-08-18)

- `lab/configs/defaults.py` → `REGISTRATION` (`align_filter: none`)
- `lab/pipeline/fringe_robust_register.py` — new `pipe-register` entry
  - writes `suite2p_cellreg/` so legacy `suite2p files/` is not overwritten
  - does **not** delete `denoised.tif`
  - diagnostics: offsets, cmax, mean original vs registered
- FFT notch **removed** from this path after clarifying defringe is upstream

Legacy `process_registration.py` is unchanged (still nonrigid on the movie
as-is, default `smooth_sigma=1.15`). Compare the two on the same stack.

## How to run

```text
python lab/pipeline/fringe_robust_register.py <parent_or_tif>
python lab/pipeline/fringe_robust_register.py <parent> --no-share-shifts
python lab/pipeline/fringe_robust_register.py path/to/stack.tif --channel A
```

Look at `suite2p_cellreg/diagnostics_register.png` and `offsets.npz`.

## Evaluation checklist (next session)

On a fringe-heavy run (LED_x15_Level1 or similar):

1. Mean `cmax` higher than ~0.01?
2. Registered mean fringes **weaker or similar**, not sharper, vs `stk_avg`?
3. If both channels: yoff correlation ChanA vs ChanB up from ~0.07?
4. Cells in a movie scrub: follow mouse motion, fringes stay put or drift slowly?
5. Weak-cmax frames interpolated, not huge junk jumps?

If residual fringes still dominate, try `align_filter=lowpass` (stronger
`lowpass_sigma` / `smooth_sigma`). Do not add an FFT notch in this repo.

## Level3b copy bakeoff (raw, independent A/B, 2026-08-18)

Sandbox: `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\`
Orchestrator: `lab/pipeline/run_mc_raw_bakeoff.py`
Compare: `lab/pipeline/compare_mc_channels.py`

Both conditions: no shared shifts, no full registered TIFFs.
`raw_legacy` = stock-like (`smooth_sigma=1.15`, nonrigid).
`raw_cell` = `REGISTRATION` (rigid, `smooth_sigma=3`, weak-cmax interpolate).

| metric | legacy | cell |
|---|---|---|
| xoff Pearson A vs B | 0.505 | **0.974** |
| yoff Pearson A vs B | 0.382 | **0.950** |
| cmax median A / B | 0.0101 / 0.0045 | 0.0035 / 0.0023 |
| ridge Fourier-y A (reg / stk_avg) | 0.113 / 0.179 | 0.139 / 0.179 |
| ridge Fourier-y B (reg / stk_avg) | **0.048 / 0.035** | **0.085 / 0.035** |
| weak frames interpolated | 0 (corrxy_frac=0) | 51 per channel |

Figures: `mc_runs/raw_legacy/compare_AB.png`, `mc_runs/raw_cell/compare_AB.png`.

Reading: cell-ops fixed the dual-channel yoff disagreement. Slow y-drift
(~+6 to −4 px) is shared, so it is more like mouse/stage than independent
fringe hopping. ChanB registered mean still shows sharper horizontal ridges
than `stk_avg` (ridge energy up; worse under cell-ops than legacy). ChanA
ridge went down in both. cmax staying ~0.01 or below is expected; do not
treat a higher cmax as success.

A high A/B correlation is necessary but not sufficient: both PMTs can lock
to the **same** electronic texture. Shared A→B (apply ChanA shifts to B)
is the next diagnostic — if A tracked cells, B fringes should blur. Also
try `align_filter=lowpass` before declaring the rigid cell profile done.

## Follow-up: share-A and lowpass (raw, 2026-08-18)

Orchestrator: `lab/pipeline/run_mc_raw_followup.py`.

**Share-A** (`mc_runs/raw_cell_shareA`): reused independent cell-ops ChanA
offsets, applied to ChanB (no re-estimate). ChanB ridge 0.085 → **0.077**
vs `stk_avg` 0.035. A's trajectory still coherently adds B stripes, so it
is not a clean cell-only lock.

**Lowpass** (`mc_runs/raw_cell_lowpass`): independent A/B, `align_filter=lowpass`
`lowpass_sigma=4`, `smooth_sigma=3`. Raw Pearson x/y 0.63 / 0.57 because
ChanB hit the ±51 px `maxregshift` cap on 43 frames. Excluding those:
x/y ~0.94 / 0.92. ChanB ridge 0.061 (still above 0.035). ChanA ridge
**0.43 vs 0.18** — worse than unfiltered cell-ops. Lowpass is not the fix
at this sigma.

## Not done yet

- CellPose / soma **weights** on the alignment movie
- Dual-channel joint estimation (currently: estimate on preferred channel, copy)
- Wiring into `basic_suite2p_walk.py` / `data_processing_master.py`
- Nonrigid after a good rigid (maybe later, small `maxregshiftNR`)
- Per-experiment YAML for align-channel and masks

## Reflections

The failure mode is not “registration is too weak”; it is **registering the
wrong layer**. Metrics that only look at residual motion in the registered
movie can look “better” while biology is worse, because lining up fringes
reduces apparent texture motion. Always compare to unregistered `stk_avg`
and to dual-channel yoff agreement.

SUPPORT before register can make residual fringes **more** coherent
(denoiser may treat them as signal). Prefer defringed-then-delivered
stacks; do not compensate by notching inside suite2p.

`fs` is a separate bug (hardcoded 10 vs ThorImage `frameRate/averageNum`).
It does not drive this fringe-lock, but record the real `fs` when we attach
a dataset.
