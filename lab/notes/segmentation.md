# Segmentation

Last updated: 2026-08-19

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
Figure: raw vs v2.1, one row per method×channel (mean | ROIs | F raster each)
→ `seg_runs/raw_vs_v21_eval/compare.png` (plus per-channel `overview.png`).
Raster is all detected ROIs, F z-scored per ROI, time in seconds.
Runner: `lab/pipeline/run_seg_eval.py` (ran 2026-08-19; see counts below).

## Raw vs v2.1 (2026-08-19), cell-ops registered movie

`seg_runs/raw_vs_v21_eval/compare.png`. Each arm is a GUI-openable `plane0`.

| | raw n ROI | v21 n ROI |
|---|---|---|
| temporal A | 229 | 109 |
| temporal B | 4 | 6 |
| cyto3 A | 502 | 469 |
| cyto3 B (wrong prior) | 28 | 14 |

## Last 2D peek (not GUI-evaluable)

CellPose `cyto3` on **unregistered** full-stack means (raw vs v2.1):
`seg_runs/cellpose_full/compare.png`. Counts barely change; motion smear
dominates. ChanB used the soma model on purpose (wrong prior).

Do detection **after** cell-oriented motion correction on a delivered
stack. Curation cannot rescue fringe-locked ROIs. Independent `v21_cell`
is the first MC condition where ChanB registered ridge did not exceed
its unregistered mean — still do not extract paper traces until
registered-movie ROIs and F/Fneu look usable.

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

1. Run `lab/pipeline/run_seg_eval.py` raw vs v21 (shared cell-ops `data.bin`
   per stack, then `temporal` vs `cyto3`). Open each `suite2p/` in both GUIs.
2. Neuron channel: suite2p temporal vs Cellpose `cyto3` on the registered
   movie (anatomical_only=2), with F and Fneu.
3. Astrocyte channel: do not reuse the neuron model; plan training data.
   Stock `cyto3` on ChanB stays a wrong-prior control.
4. Keep `ops['anatomical_only']` / `lab/configs` CELLPOSE + SEG_EVAL as
   the hook.

Paper-repo: `catalog/preprocessing/TODO.md` (CellPose section),
`catalog/roi_trace/OVERVIEW.md`.
