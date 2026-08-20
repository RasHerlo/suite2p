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

ChanA/ChanB are PMT paths. Cell type depends on prep + filter cube (Shinano
C1: ChanA red neurons, ChanB green astro). Do not hardcode globally.

## When we pick this up

1. Run `lab/pipeline/run_seg_eval.py` raw vs v21 vs v22 (shared cell-ops
   `data.bin` per stack, then `temporal` vs `cyto3`). Open each `suite2p/`
   in both GUIs. **Done 2026-08-20** (`raw_vs_v21_vs_v22_eval`).
2. Neuron channel: suite2p temporal vs Cellpose `cyto3` on the registered
   movie (anatomical_only=2), with F and Fneu.
3. Astrocyte channel: do not reuse the neuron model; plan training data
   (or an existing astrocyte Cellpose model). Stock `cyto3` on ChanB stays
   a wrong-prior control.
4. Keep `ops['anatomical_only']` / `lab/configs` CELLPOSE + SEG_EVAL as
   the hook.

Paper-repo: `catalog/preprocessing/TODO.md` (CellPose section),
`catalog/roi_trace/OVERVIEW.md`.
