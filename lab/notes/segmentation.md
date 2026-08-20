# Segmentation

Last updated: 2026-08-20

## Evaluation contract (from 2026-08-19)

A segmentation comparison is **detection + trace extraction**, not a 2D
mask overlay. Each arm must write a real `suite2p/plane0` so it can be
opened in:

1. This repo’s **suite2p GUI** (load `plane0/stat.npy`)
2. [s2p_Trace_Curation](https://github.com/RasHerlo/s2p_Trace_Curation)
   (File → Open suite2p folder… = the folder that **contains** `plane0/`)

Required in `plane0/`: `ops.npy`, `stat.npy`, `F.npy`, `Fneu.npy`,
`iscell.npy`, `data.bin`. OASIS / spike deconvolution is **off**
(`spikedetect=False`). Suite2p still writes zeros to `spks.npy` because
the suite2p GUI loader rejects a folder that lacks it.

Do **not** apply lab ellipticity ROI filtering on these runs — evaluate
what detection actually produced. Neuropil extraction stays on
(`neuropil_extract=True`). Keep `data.bin` (`delete_bin=False`); both
GUIs need the registered movie.

Layout: `seg_runs/<kind>_cell_<method>/ChanA|B/suite2p/plane0/`
Figure: one row per method×channel, one 3-panel block per kind
(mean | ROIs | F raster)
→ `seg_runs/raw_vs_v21_vs_v22_eval/compare.png` (plus per-channel `overview.png`).
Older two-way figure remains at `seg_runs/raw_vs_v21_eval/`.
Raster is all detected ROIs, F z-scored per ROI, time in seconds.
Runner: `lab/pipeline/run_seg_eval.py`.

## Raw vs v2.1 vs v2.2 (2026-08-20), cell-ops registered movie

Stacks: `inputs/raw`, `inputs/defringed_v21`, `inputs/defringed_v22`
(5400×512×512). v22 is pack_D seeded-500; promoted tiffs, not the
`defringe_runs/` diagnostics folder. Shared cell-ops `data.bin` per
kind×channel under `seg_runs/_bin/`, then `temporal` vs `cyto3`.

`seg_runs/raw_vs_v21_vs_v22_eval/compare.png`. Each arm is a GUI-openable `plane0`.

| | raw n ROI | v21 n ROI | v22 n ROI |
|---|---|---|---|
| temporal A | 229 | 109 | 113 |
| temporal B | 4 | 6 | 6 |
| cyto3 A | 502 | 469 | 490 |
| cyto3 B (wrong prior) | 28 | 14 | 22 |

v22 tracks v21, not raw, on **temporal** ChanA (113 vs 109 vs 229). cyto3
ChanA sits between v21 and raw (490 vs 469 vs 502). Temporal ChanB stays
essentially empty. cyto3 ChanB is still the soma model on astrocytes.

ChanA F rasters keep a common ~180 s vertical band across raw/v21/v22
(physiology or a shared acquisition event — not removed by defringe).

Existing `plane0` folders from the independent-B bakeoff (2026-08-20) are
still valid for that comparison. New `--overwrite` runs use share-A.

## Last 2D peek (not GUI-evaluable)

CellPose `cyto3` on **unregistered** full-stack means (raw vs v2.1):
`seg_runs/cellpose_full/compare.png`. Counts barely change; motion smear
dominates. ChanB used the soma model on purpose (wrong prior).

Do detection **after** cell-oriented motion correction on a delivered
stack. Curation cannot rescue fringe-locked ROIs. Default MC is share-A;
use `ChanB/independent_meanImg.png` when judging residual fringes. Do not
extract paper traces until registered-movie ROIs and F/Fneu look usable.

## Two families (not one model)

| Path | What it uses | Fits |
|---|---|---|
| Suite2p sparse/temporal | pixel covariance over time | functional ROIs; can glue to fringes or neuropil |
| CellPose spatial | trained shape prior on mean/max image | compact **neuronal somata** (`cyto` / `cyto3` worth trying) |

Stock Cellpose is **not** an in-vivo astrocyte model. Territories are bushy,
overlapping G-Flamp1 clouds. On `acq-260511-led-level1`, suite2p already
chopped ChanB into ~4 µm blobs (3.4% FOV) while clouds are much larger.
Expect stock Cellpose to do the same unless we train (or adopt) an astrocyte
model.

**ChanA/ChanB are PMT paths.** Cell type depends on microscope + filter cube:

| Rig | `Experiment.xml` `<Computer>` | Green / G-Flamp astro | Red / jRGECO-RCaMP neuron | `align_channel` |
|---|---|---|---|---|
| Shinano | `THORLABS_30_016` | ChanB | ChanA | A |
| Musashi | `USER-PC` | ChanA | ChanB | B |

Do not hardcode globally. This batch: 260510–260521 = Shinano; **260616 = Musashi**.

## When we pick this up

1. **Locked default:** run **both** `temporal` and `cyto3` (anatomical) on
   the same share-A `data.bin`. Intercalation stays in s2p_Trace_Curation.
   Runner: `lab/pipeline/run_seg_locked.py`.
2. Astrocyte temporal: `tau=1 s`, correlation window **5.61 µm** (suite2p
   `spatial_scale=1` / 6 px at 0.935 µm/px).
3. Astrocyte anatomical: `cyto3`, `anatomical_only=2`, soma diameter
   **8.42 µm** (9 px at 0.935 µm/px), `flow_threshold=0.4`. Not a
   territory model. Custom weights can still replace `pretrained_model`.
4. Both convert from `Experiment.xml` (`fs = frameRate/averageNum`,
   `um_per_px = LSM/@pixelSizeUM`). Do not hardcode 9 px or scale=1 on a
   new FOV. Neurons: temporal auto-scale; anatomical cyto3 diameter auto.
5. Sandbox bakeoff `raw_vs_v21_vs_v22_eval` remains the Level3b reference
   (`seg_runs/` in the Level3b copy). ChanB cyto3 there used auto diameter
   (wrong prior); the locked AC arm is the 2026-08-20 `seg_cyto3_d9` check.

## Level3b ChanB tau sweep (2026-08-20)

Share-A v22 movie:
`mc_runs\260511\...\LED_x15_Level3b\seg_tau_sweep\compare.png`

`spatial_scale` locked at **2** (12 px) so tau is the only knob. `high_pass`
scaled to keep a ~100 s window. OASIS off.

| tau (s) | bin frames | n ROI | coverage |
|---|---|---|---|
| 1 | 15 | 9 | 0.009 |
| 3 | 44 | 0 | 0 |
| 5 | 74 | 0 | 0 |
| 10 | 148 | 0 | 0 |

Slower bins emptied detection; they did not recover G-Flamp somata. A
side run with auto-scale at tau=3 found 12 ROIs at **6 px** before the
locked-scale rerun. Next knob is `spatial_scale` 1 vs 2 vs 3 at tau=1
(and maybe threshold_scaling), not a wider tau grid. Then Cellpose.

## Level3b ChanB spatial_scale sweep (2026-08-20)

Same share-A movie, `tau=1`. Figure:
`mc_runs\260511\...\LED_x15_Level3b\seg_spatial_scale_sweep\compare.png`

| scale | px (~µm at 0.935) | n ROI | coverage |
|---|---|---|---|
| 1 | 6 (~5.6) | 359 | 0.087 |
| 2 | 12 (~11) | 9 | 0.009 |
| 3 | 24 (~22) | 0 | 0 |

Finer scale over-segments; 24 px finds nothing. Temporal sparse detect is
not locking onto soma-sized objects here.

**Locked for future astrocyte temporal runs (2026-08-20):** `tau=1 s`,
correlation window **5.61 µm** (`spatial_scale=1` / 6 px at the lock FOV
0.935 µm/px) on **astrocytes**, wherever they sit.
`TEMPORAL_BY_CELL_TYPE["astrocyte"]`. XML `pixelSizeUM` retunes the
suite2p scale (nearest of 6/12/24/48 px). Which PMT: experiment
`CHANNEL_CELL_TYPES` / `cell_type_by_channel` if sensors moved, else
MICROSCOPES cube defaults (Shinano ChanB, Musashi ChanA with current
G-Flamp/jRGECO). Neurons stay `tau=1`, `spatial_scale=0` (auto).
Share-align follows the **neuron** PMT the same way.

**Locked for astrocyte anatomical (2026-08-20):** `cyto3` on the share-A
mean (`anatomical_only=2`), soma diameter **8.42 µm** (9 px at 0.935 µm/px),
`flow_threshold=0.4`. Check: `mc_runs\260511\...\seg_cyto3_d9\compare.png`
(22–38 ROIs / FOV on ChanB). `ANATOMICAL_BY_CELL_TYPE["astrocyte"]`.
Same XML conversion. Neurons: cyto3, diameter auto, flow 1.5 until locked.

Default pipeline writes **both** arms; curation intercalates them.

Paper-repo: `catalog/preprocessing/TODO.md` (CellPose section),
`catalog/roi_trace/OVERVIEW.md`.
