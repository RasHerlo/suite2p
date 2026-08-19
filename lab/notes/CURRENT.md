# Current work

Last updated: 2026-08-19

**Paper / catalog agent:** [HANDOFF_FOR_PAPER_REPO.md](HANDOFF_FOR_PAPER_REPO.md)
(not this scratchpad).

**Active:** independent cell-ops MC on full-stack **v2.1**. Next seg test must
write GUI-openable `suite2p/plane0` (detection + F/Fneu, no deconv).

**Do not:** re-implement defringe here; turn on `1Preg`; extract Fig 1 traces
yet; use lowpass registered means as Fig 1 stills (use `stk_avg`).

**v2.1 full stacks (2026-08-18 evening):**
`inputs/defringed_v21/ChanA|B/*_stk_defringed_v21.tif` (5400×512×512).

**MC on v21** (ridge vs *defringed* unregistered mean, not raw `stk_avg`):

| | xoff r | yoff r | ChanA ridge reg vs unreg | ChanB ridge reg vs unreg |
|---|---|---|---|---|
| `raw_cell` (ref) | 0.97 | 0.95 | 0.139 vs 0.179 | **0.085 vs 0.035** |
| `v21_cell` | 0.98 | 0.93 | 0.143 vs 0.452 | **0.057 vs 0.072** |
| `v21_cell_shareA` | 1.00* | 1.00* | 0.143 vs 0.452 | **0.076 vs 0.034** |

Independent v21: first time ChanB registered ridge **does not exceed** its unregistered mean. A/B still agree. Share-A still raises B ridge (and B unreg baseline does not match the independent-B mean — treat share-A ridge_avg_B with care).

Figures: `mc_runs/v21_cell/compare_AB.png`, `mc_runs/v21_cell_shareA/compare_AB.png`.

**CellPose `cyto3` on unregistered full-stack means** (no MC; ChanB = wrong prior):

| | n ROI | coverage | med ecc |
|---|---|---|---|
| raw A / v21 A | 122 / 116 | 0.193 / 0.181 | 0.89 / 0.91 |
| raw B / v21 B | 12 / 11 | 0.051 / 0.050 | 0.89 / 0.91 |

A vs B mask overlap 0.05–0.08 (not a shared stripe field). Counts barely change: motion smear in the unregistered mean dominates CellPose. Overlay: `seg_runs/cellpose_full/compare.png`.

**Next:** `python lab/pipeline/run_seg_eval.py` — temporal vs `cyto3` on the
v21 cell-ops registered movie, each arm a `suite2p/plane0` with F and Fneu.
Compare figure: registered mean | ROIs | F raster
(`seg_runs/v21_cell_eval/compare.png`). Evaluate in this repo’s suite2p GUI
and in s2p_Trace_Curation. Do not extract paper traces yet. The 2026-08-18
`cellpose_full` peek is not that deliverable.

**Scope:** this repo does **not** defringe. Stacks arrive already processed
upstream if needed ([derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise)).
Here: motion correction, then segmentation.

**Read next:**

- [notes/motion_correction.md](motion_correction.md)
- [pipeline/fringe_robust_register.py](../pipeline/fringe_robust_register.py)
- [configs/defaults.py](../configs/defaults.py) → `REGISTRATION`

**Sandbox (2026-08-18):**
`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\`
- `Experiment.xml` (fs = 29.595/2 = **14.80 Hz**)
- `inputs/raw`, `inputs/defringed`, `inputs/support`
- MC outputs: `mc_runs/` (do not write into the original DATA tree)

**Last decision:** No FFT notch in the register path. Register the delivered
movie with cell-oriented ops (`smooth_sigma=3`, `1Preg=False`, rigid-only).
Optional `align_filter=lowpass` is a phasecorr weighting, not a defringe
product. One shift trace can be shared across ChanA/B (same input kind only;
SUPPORT is 60 frames shorter).

**Bakeoff (2026-08-18), independent raw ChanA/B on Level3b copy:**

| | xoff r | yoff r | cmax A/B | ridge A reg vs avg | ridge B reg vs avg |
|---|---|---|---|---|---|
| `mc_runs/raw_legacy` | 0.51 | 0.38 | 0.010 / 0.0045 | 0.113 vs 0.179 | **0.048 vs 0.035** |
| `mc_runs/raw_cell` | **0.97** | **0.95** | 0.0035 / 0.0023 | 0.139 vs 0.179 | **0.085 vs 0.035** |

Cell-ops: A and B agree on motion (the original yoff ~0.07 failure is gone).
ChanB registered mean still sharpens Fourier-y ridges vs `stk_avg` (worse
than legacy on that metric). Figures: `mc_runs/raw_*/compare_AB.png`.

**Follow-up (2026-08-18), still raw Level3b copy:**

| | xoff r | yoff r | ChanB ridge reg vs avg | notes |
|---|---|---|---|---|
| `raw_cell` (independent) | 0.97 | 0.95 | **0.085 vs 0.035** | baseline |
| `raw_cell_shareA` | 1.00* | 1.00* | **0.077 vs 0.035** | *B uses A's shifts |
| `raw_cell_lowpass` | 0.63 (0.94 excl. 43 jumps) | 0.57 (0.92 excl.) | **0.061 vs 0.035** | ChanA ridge **0.43 vs 0.18**; B hit ±51 px cap |

Share-A barely lowered ChanB ridge, so A's trace also lines up B stripes (not a clean cell-only lock).
Lowpass did not fix fringe promotion and broke B (outlier y-shifts).
Lowpass ChanA means look crisp — that is **not** the MC winner (ChanA Fourier-y ridge 0.18 → 0.43).
