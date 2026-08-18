# Handoff: suite2p repo → paper / catalog repo

**Audience:** an agent or person working in
[figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper)
(catalog, Fig 1 traces, preprocessing diagnosis).

**This file is the entry point.** Read it before `CURRENT.md` or the bakeoff
figures. Do not edit MouseLand `suite2p/` or re-implement defringe here.

Last updated: 2026-08-18 (evening).

**Status (read this first):**

- Full-stack **v2.1** is promoted:
  `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\inputs\defringed_v21\`
- Independent cell-ops MC on that stack (`mc_runs/v21_cell`) is the first
  condition where **ChanB registered ridge does not exceed** the defringed
  unregistered mean. A/B yoff still agrees (~0.93). Share-A is **not** a pass.
- Raw MC (`raw_cell`) is **not** safe for traces (ChanB ridges sharpen).
- **Do not extract paper traces yet.** CellPose on unregistered means barely
  changed (motion smear). Next here: CellPose on *registered* means, then
  temporal suite2p.
- Fig 1 **stills:** keep assembled `stk_avg`. Do **not** use
  `raw_cell_lowpass` registered means (pretty, but ChanA ridge 0.18 → 0.43).

| Clone (local) | `C:\Users\rasmu\Projects\Repos\suite2p` |
|---|---|
| GitHub | https://github.com/RasHerlo/suite2p |
| Interpreter | `C:\Users\rasmu\anaconda3\envs\suite2p\python.exe` (not WindowsApps `python`) |
| Deeper notes | [CURRENT.md](CURRENT.md), [motion_correction.md](motion_correction.md), [segmentation.md](segmentation.md) |

If this path is missing on GitHub `main`, use the **local clone** — lab notes
and MC scripts may be ahead of the last push.

---

## Three-repo split (do not collapse)

| Repo | Job | Must not do |
|---|---|---|
| [derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise) | Remove PMT/scan fringes (leading: **v2.1** raw adaptive). Optional SUPPORT is downstream of that repo’s README, not here. | Motion correction, ROI detection, paper traces |
| **This repo (suite2p)** | Motion correction, then segmentation / traces **on a delivered stack** | FFT-notch / v2.1 defringe; `1Preg`; writing into the original session `DATA/` tree |
| [figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper) | Catalog, figures, decide where the pipeline still fails | Copy pipeline code; treat fringe-locked `iscell` as biology |

Sandbox (shared data, not git):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

Layout and promotion rules: `README_SANDBOX.md` in that folder.

| Stage | Writes |
|---|---|
| Defringe trials | `defringe_runs/` (underscore) |
| SUPPORT trials | `support_runs/` |
| Motion correction (this repo) | `mc_runs/` |
| Segmentation peeks (this repo) | `seg_runs/` |
| Joint pipelines | `combo_runs/` |

v2.1 full stacks **are promoted** to `inputs/defringed_v21/`.
`inputs/defringed/` is older **v2** (ChanA weak `q≈6`) — not the candidate.

---

## Dataset (this bakeoff ≠ the paper FOV)

| | |
|---|---|
| Sandbox source | Shinano `LED_x15_Level3b` copy (`SOURCE.txt` in the sandbox) |
| Paper diagnosis FOV | `acq-260511-led-level1` (same animal/day family, **different** LED level / FOV) |
| `fs` | ThorImage `frameRate/averageNum` = 29.595/2 = **14.80 Hz** (not hardcoded 10) |
| Size | raw 5400×512×512 uint16; SUPPORT `denoised_cut` is **5340** frames (30+30 cut) |
| Channels | PMT paths, not cell types. Shinano + C1: **ChanA = red neurons (jRGECO1a)**, **ChanB = green astro (G-Flamp1)** |

Never share shift traces between raw/defringed (5400) and SUPPORT (5340).

---

## Progress (2026-08-18) — motion correction on **raw**

Code (run from suite2p repo root):

- `lab/pipeline/fringe_robust_register.py` — cell-oriented register (`REGISTRATION` in `lab/configs/defaults.py`)
- `lab/pipeline/compare_mc_channels.py` — A vs B shifts, `cmax`, Fourier-y ridge vs `stk_avg`
- `lab/pipeline/run_mc_raw_bakeoff.py` — independent raw A/B, legacy vs cell
- `lab/pipeline/run_mc_raw_followup.py` — share ChanA shifts onto B; independent lowpass

Ops that matter: `smooth_sigma=3`, `nonrigid=False`, `1Preg=False`, weak-`cmax`
interpolation (`corrxy_frac=0.4`). **No FFT notch.** Optional `align_filter=lowpass`
is phasecorr weighting only; the kept movie is still the delivered intensities.

### Independent raw A/B

| condition | xoff r | yoff r | cmax A/B | ridge A (reg / `stk_avg`) | ridge B (reg / `stk_avg`) |
|---|---|---|---|---|---|
| `mc_runs/raw_legacy` | 0.51 | 0.38 | 0.010 / 0.0045 | 0.113 / 0.179 | **0.048 / 0.035** |
| `mc_runs/raw_cell` | **0.97** | **0.95** | 0.0035 / 0.0023 | 0.139 / 0.179 | **0.085 / 0.035** |

Legacy-like (`smooth_sigma=1.15`, nonrigid) still disagrees in y (the fringe
axis). Cell-ops **agree** across PMTs. That is necessary, not sufficient.

### Follow-ups (still raw)

| condition | ChanB ridge (reg / avg) | reading |
|---|---|---|
| `raw_cell` independent | 0.085 / 0.035 | B registered mean sharpens stripes |
| `raw_cell_shareA` (A shifts → B) | 0.077 / 0.035 | A’s trace also lines up B stripes |
| `raw_cell_lowpass` (σ=4) | 0.061 / 0.035 | ChanA ridge **0.43 / 0.18**; B hit ±51 px cap on 43 frames |

After a 2 s smooth of `raw_cell` traces: slow A vs B x/y **r = 0.999**. Fast y
residual r ≈ **0.31** (independent fringe jitter). Share-A still raised B ridge,
so the **slow common-mode drift** is enough to promote B stripes.

Figures (open these):

```
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\mc_runs\raw_legacy\compare_AB.png
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\mc_runs\raw_cell\compare_AB.png
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\mc_runs\raw_cell_shareA\compare_AB.png
F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\mc_runs\raw_cell_lowpass\compare_AB.png
```

JSON next to each PNG. Per-channel: `ChanA|B/offsets.npz`, `ops.npy`,
`diagnostics_register.png`. Full registered TIFFs were **not** written (disk).

Ridge metric: fraction of 2D-FFT power at `|ky| > 0.05` (~finer than 20 px in y)
on the **mean image**. Horizontal PMT ridges live in Fourier-y. Registered ridge
**must not exceed** `stk_avg`. A falling ChanA ratio can be sharper somata in
the denominator; a **rising ChanB ratio** means stripes added coherently.

---

## Warnings (paper-repo must treat as hard)

1. **Do not extract Fig 1 (or any paper) traces from fringe-locked `iscell`.**
   Curation cannot rescue stripe-shaped ROIs.
2. **Do not treat a still registered movie as success.** Lining up fringes also
   looks still. Always compare registered mean vs `stk_avg` and dual-channel yoff.
3. **Do not treat high A/B shift correlation as a cell lock.** Both PMTs can
   track the same electronic texture. Share-A is the test: B ridges should
   *blur* if A followed tissue.
4. **Do not turn on `1Preg` / `spatial_hp_reg=42`.** That high-pass keeps ~10 px
   (fringe band) and removes large astrocyte clouds.
5. **Do not FFT-notch inside this repo.** Defringe is
   [derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise)
   (v2.1: `reference/gpt/pmt_fringe_raw_adaptive_v21.py`).
6. **Do not write MC outputs into the original `LED_x15_Level3b\DATA` tree.**
   Only the sandbox `mc_runs/`.
7. **Do not share shifts across unequal lengths** (SUPPORT cut vs full stack).
8. **Do not use stock Cellpose on ChanB astrocytes.** Soma prior; clouds get
   chopped into ~4 µm blobs (seen on level1). Neuron vs astrocyte models must
   differ. ChanA/B are PMT letters, not a global cell-type map.
9. **Do not use `inputs/defringed/` (v2) as the delivered stack.** Use
    `inputs/defringed_v21/` (full 5400-frame v2.1).
10. **Do not judge MC by median `cmax` going up**, or by a crisp registered
    mean. Lowpass ChanA looks sharp and **fails** the ridge test (0.18 → 0.43).
    Fig 1 stills stay on `stk_avg` until a registered mean passes ridge **and**
    eye QC.
11. **Do not extract traces from `v21_cell` yet.** ChanB ridge pass is
    necessary, not sufficient (share-A still raises B ridge; no temporal
    ROIs yet).
12. Legacy master order
    `SUPPORT → register → defringe` **feeds phasecorr the worst texture.**
    Preferred: `assemble → defringe v2.1 → (optional SUPPORT) → register → segment`.
13. Older walkers hardcoded `fs=10`. Record the `fs` actually used when
    attaching a dataset.

---

## Reflections

The failure is **wrong-layer registration**, not weak registration. Phase
correlation **whitens** Fourier amplitude, so a periodic PMT texture can win
the peak even when dim. `smooth_sigma` / lowpass only change what the matcher
sees; they do not remove stripes from the movie.

On **raw**, cell-ops is the least-bad recipe we ran (A/B agreement). It is
**not** a clean cell lock: applying those ChanA shifts still sharpens ChanB
ridges. Extra spatial lowpass made it worse. Remaining cheap ops
(`smooth_sigma` 4–5, CellPose-weighted alignment, leave B unregistered) are
not expected to make raw ChanB safe for ROIs. Unregistering B also breaks a
shared neuron/astro coordinate frame.

SUPPORT **before** defringe can make residual fringes *more* coherent (denoiser
may treat them as signal). Prefer a delivered defringed stack, then MC.

Paper FOV `led_x15_level1` showed the same class of failure (yoff r ~0.07,
`corrXY` ~0.01/0.005, SUPPORT iscell ~76% FOV fringe masks on ChanA). Level3b
numbers differ; the mechanism does not.

---

## Forward plan (coordination)

### This repo (v2.1 full stacks, 2026-08-18 evening)

Promoted: `inputs/defringed_v21/ChanA|B/*_stk_defringed_v21.tif` (5400 frames).

**MC** (`mc_runs/v21_cell`, cell-ops independent; ridge vs *defringed*
unregistered mean):

- A/B xoff r=0.98, yoff r=0.93 (similar to raw_cell).
- ChanB registered ridge **0.057 vs 0.072 unreg** — first time B does not
  promote ridges. Raw_cell was 0.085 vs 0.035 (fail).
- ChanA 0.143 vs 0.452 (fraction; v21 unreg mean has a large ky share).
- Share-A (`v21_cell_shareA`) still raises B ridge; do not treat as pass.

**CellPose cyto3** on unregistered means (`seg_runs/cellpose_full/`):
raw vs v21 almost the same (ChanA 122 vs 116 ROIs). Motion smear dominates.
ChanB 11–12 ROIs with soma model (wrong prior). Overlay `compare.png`.

Not done: CellPose on *registered* means; temporal suite2p; paper traces.

### Defringe repo

Full-stack v2.1 delivered. Further knobs live in that repo.

### Paper / catalog repo (what to record, not implement)

- Pin `pipeline_id` + **this** suite2p commit + defringe v2.1 commit + `fs` +
  `order: defringe_first` on `catalog/preprocessing.yaml` when a dataset is
  attached. Sandbox Level3b is **not** the paper FOV (`led_x15_level1`).
- FOV stills: assembled `stk_avg` until registered-mean QC is signed off.
- ROI/trace tools stay downstream. See `catalog/preprocessing/TODO.md` and
  `catalog/roi_trace/OVERVIEW.md`.

### Next in this repo (not paper traces)

1. CellPose on **registered** means (`raw_cell` vs `v21_cell`).
2. Temporal suite2p detection on v21 (after that overlay looks like cells).
3. Astrocyte model / not stock `cyto3` on ChanB.

### Explicitly not next (raw)

Do not spend more raw-MC cycles expecting paper-grade ROIs. Do not put
`raw_cell_lowpass` means in Fig 1.

---

## How to inspect without re-running

1. This file (status + warnings + plan).
2. Sandbox `README_SANDBOX.md`.
3. MC: `mc_runs/raw_cell/compare_AB.png`, `mc_runs/v21_cell/compare_AB.png`
   (+ `.json`). Lowpass is a negative example:
   `mc_runs/raw_cell_lowpass/compare_AB.png`.
4. CellPose: `seg_runs/cellpose_full/compare.png` + `metrics.json`.
5. `lab/notes/motion_correction.md` for ops rationale.

Do not start a new MC or segment run from the paper repo. Ask this repo if
the next named test is needed.
