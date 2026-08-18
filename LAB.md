# Lab pipeline

This repository is a **suite2p fork** plus lab-specific processing. Upstream
MouseLand code stays in `suite2p/`. Everything we own lives under `lab/`.

Run commands from the repository root (or `pip install -e .` so `lab` is
importable). Either form works:

```text
python -m lab.pipeline.basic_suite2p_walk <root_dir>
python lab/pipeline/basic_suite2p_walk.py <root_dir>
```

## Layout

```text
suite2p/                 upstream engine — do not edit unless ops/wrappers are not enough
lab/
  configs/               shared defaults + experiment config template
  preprocess/            FFT, wavelet, video denoise, ripple removal
  pipeline/              batch walkers, master orchestration, registration helpers
  detection/             ROI filters and manual selector
  postprocess/           traces, rasterplots, pickle export, PCA
  cellpose/              model path registry (neuron / astrocyte)
  diagnostics/           compare suite2p output folders
  scratch/               one-off / hardcoded-path scripts, not production
models/                  Cellpose weights (gitignored)
scripts/                 upstream suite2p helper scripts (not lab)
```

## Pipeline map

```text
raw / SUPPORT stacks
        │
        ▼
  lab/preprocess     optional denoise (FFT GUI, FFT masks, wavelet, video, ripple)
        │
        ▼
  lab/pipeline       suite2p registration → detection → extraction
                     (ops from lab/configs/defaults.py)
        │
        ▼
  lab/detection      ellipticity / connected-component filter, or manual review
        │
        ▼
  lab/postprocess    rasterplots, pickle traces, stimulation / PCA plots
```

Cellpose is not on this path yet. `lab/configs/defaults.py` already has a
`CELLPOSE` slot (`anatomical_only: 0`). Point `models.neuron` /
`models.astrocyte` at weights in `models/` and raise `anatomical_only` when
you want anatomical detection.

## What to run

| Task | Script |
|---|---|
| Interactive FFT mask | `lab/preprocess/DAO_FFT.py` |
| Apply FFT masks from CLI | `lab/preprocess/FFT_mask_conversion.py` |
| Other denoise | `wavelet_denoise.py`, `denoise_video.py`, `ripple_remove.py` in `lab/preprocess/` |
| Batch suite2p + ROI filter | `lab/pipeline/basic_suite2p_walk.py` |
| Older FFT → s2p → ROI → raster → pickle | `lab/pipeline/data_processing_master.py` |
| Registration only (SUPPORT folders) | `lab/pipeline/process_registration.py` |
| Extract registered stack from `data.bin` | `lab/pipeline/extract_registered_stack.py` |
| Single-step ROI / raster / pickle | `lab/pipeline/run_s2p_functions.py --list` |
| Manual ROI review | `lab/detection/manual_roi_selector.py` |
| Compare two suite2p folders | `lab/diagnostics/s2p_settings_comparison.py` |
| Trace PCA / stim plots | `lab/postprocess/post_process_neus.py` |

## Status of scripts

**Production-ish** (used in walkers / master): `basic_suite2p_walk.py`,
`data_processing_master.py`, `roi_selection_new.py`, `run_roi_selection.py`,
`run_rasterplots.py`, `generate_traces_pickle.py`, `FFT_mask_conversion.py`,
`DAO_FFT.py`.

**Useful tools**: `manual_roi_selector.py`, `process_registration.py`,
`extract_registered_stack.py`, `s2p_settings_comparison.py`,
`post_process_neus.py`, the other denoisers.

**Scratch** (`lab/scratch/`): `extract_meanImg_temp.py` (hardcoded path),
`FFT_examples.py`.

## Defaults

Shared numbers live in `lab/configs/defaults.py`:

- suite2p ops (`fs`, `tau`, nonrigid, neuropil, …)
- ROI ellipticity / components thresholds
- channel folder → tiff names
- FFT masks per channel A/B
- stimulation windows by `F.npy` length
- Cellpose model placeholders

`lab/configs/experiment.example.yaml` is the shape for per-experiment
overrides. It is not loaded yet.

## Rule of thumb

Change **lab/** for new lab behavior. Change **suite2p/** only when a wrapper
and `ops` cannot do the job (for example merging two Cellpose mask sets on
one plane).
