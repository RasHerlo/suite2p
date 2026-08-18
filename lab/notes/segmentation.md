# Segmentation

Last updated: 2026-08-18 (evening)

CellPose `cyto3` peek on **unregistered** full-stack means (raw vs v2.1):
`seg_runs/cellpose_full/compare.png`. Counts barely change; motion smear
dominates. ChanB used the soma model on purpose (wrong prior).

Do this **after** cell-oriented motion correction on a delivered stack.
Curation cannot rescue fringe-locked ROIs. Independent `v21_cell` is the
first MC condition where ChanB registered ridge did not exceed its
unregistered mean — still do not extract paper traces until overlays on
**registered** means look like cells, and temporal detection is checked.

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

1. Confirm register no longer promotes fringes.
2. Neuron channel: suite2p temporal vs Cellpose `cyto3` on mean/max.
3. Astrocyte channel: do not reuse the neuron model; plan training data.
4. Keep `ops['anatomical_only']` / `lab/configs` CELLPOSE slot as the hook.

Paper-repo: `catalog/preprocessing/TODO.md` (CellPose section),
`catalog/roi_trace/OVERVIEW.md`.
