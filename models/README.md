# Cellpose models

Put trained weights here. They are gitignored because the files are large.

Suggested names (match `lab/configs/defaults.py` → `CELLPOSE['models']`):

- `neuron` — Cellpose model for neurons (can start from the built-in `cyto` / `cyto3`)
- `astrocyte` — custom astrocyte model, once trained

Until those paths are set, suite2p keeps using functional ROI detection
(`anatomical_only: 0`).
