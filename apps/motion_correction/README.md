# Suite2p Motion Correction (GUI)

Motion-corrects one TIFF stack (either channel) with the stock Suite2p registration
(`registration_wrapper`: reference image, rigid + optional non-rigid). No defringe or
two-channel logic.

**Outputs** (in the chosen folder): `<name>_registered.tif`, `<name>_shifts.csv`, `<name>_meanImg.tif`.

`shifts.csv` columns: `frame, yoff, xoff, corrXY, badframe` and, if non-rigid is on,
`yoff_blockN` / `xoff_blockN` per block. Sign convention is Suite2p's.
Registered output is int16 (uint16 input is halved to fit, as in Suite2p).

Run from source: `python apps/motion_correction/app.py` (needs numpy, scipy, numba, torch, tifffile).

Build the .exe: `powershell -File apps/motion_correction/build_exe.ps1` (needs `pip install pyinstaller`),
then zip `dist/Suite2pMotionCorrection/` and attach it to a GitHub Release.
