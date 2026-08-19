# Current work

Last updated: 2026-08-19 (late)

**Paper / catalog agent:** [HANDOFF_FOR_PAPER_REPO.md](HANDOFF_FOR_PAPER_REPO.md)
(not this scratchpad). SUPPORT agent: [HANDOFF_FOR_SUPPORT.md](HANDOFF_FOR_SUPPORT.md).

**Active:** signature-mask **cell vs fringe** scores are the default MC
metrics (`lab/pipeline/mc_fft_metrics.py`). Honest rescore of `raw_cell`
(re-registered), `v21_cell`, `v21_cell_shareA`: all `both_up`. Seg bakeoff
still waiting GUI/s2p_Trace_Curation look. Do not extract paper traces yet.

**Do not:** re-implement defringe here; turn on `1Preg`; extract Fig 1 traces
yet; use lowpass registered means as Fig 1 stills (use `stk_avg`).

**v2.1 full stacks (2026-08-18 evening):**
`inputs/defringed_v21/ChanA|B/*_stk_defringed_v21.tif` (5400×512×512).

**MC scores (2026-08-19 late), reg/unreg power, v2.1 `signature.json`:**

| | xoff r | yoff r | cell A/B | fringe A/B |
|---|---|---|---|---|
| `raw_cell` | 0.974 | 0.950 | 1.95 / 1.34 | 1.88 / **10.5** |
| `v21_cell` | 0.975 | 0.935 | 1.95 / 1.33 | 1.90 / **13.0** |
| `v21_cell_shareA` | 1.00* | 1.00* | 1.95 / 1.35 | 1.90 / **3.91** |

Cell band matches the crispier registered means. PMT family also rises
(independent B badly). Share-A cuts B-family freeze vs estimating B but
does not pass. Legacy `|ky|>0.05` ridge still rises (A ~0.18→0.45, B
~0.034→0.07) and cannot see the share-A vs independent B family gap.
ChanA fringe *fraction* can fall while absolute family power is still
~1.9× — use power ratios. Figures: `mc_runs/<run>/compare_AB.png`.

**Seg+extraction bakeoff (2026-08-19), cell-ops registered movie, F/Fneu, no OASIS:**

| | raw n ROI | v21 n ROI |
|---|---|---|
| temporal A | 229 | 109 |
| temporal B | 4 | 6 |
| cyto3 A | 502 | 469 |
| cyto3 B (wrong prior) | 28 | 14 |

v21 temporal ChanA ~halves ROI count vs raw. cyto3 ChanA stays dense (~500).
Temporal ChanB still almost empty. Figure: `seg_runs/raw_vs_v21_eval/compare.png`.
`plane0` folders: `seg_runs/<kind>_cell_<method>/ChanA|B/suite2p/`.

**CellPose `cyto3` on unregistered full-stack means** (no MC; ChanB = wrong prior):

| | n ROI | coverage | med ecc |
|---|---|---|---|
| raw A / v21 A | 122 / 116 | 0.193 / 0.181 | 0.89 / 0.91 |
| raw B / v21 B | 12 / 11 | 0.051 / 0.050 | 0.89 / 0.91 |

A vs B mask overlap 0.05–0.08 (not a shared stripe field). Counts barely change: motion smear in the unregistered mean dominates CellPose. Overlay: `seg_runs/cellpose_full/compare.png`.

**Next:** inspect `seg_runs/raw_vs_v21_eval/compare.png` and open `plane0`
folders in this repo’s suite2p GUI and in s2p_Trace_Curation. Do not extract
paper traces yet. The 2026-08-18 `cellpose_full` peek is not that deliverable.

**Scope:** this repo does **not** defringe. Stacks arrive already processed
upstream if needed ([derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise)).
Here: motion correction, then segmentation.

**Read next:**

- [notes/motion_correction.md](motion_correction.md)
- [pipeline/mc_fft_metrics.py](../pipeline/mc_fft_metrics.py)
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
