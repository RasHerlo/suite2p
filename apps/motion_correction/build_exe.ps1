# Build a single-folder Windows app. Run from anywhere:  powershell -File apps\motion_correction\build_exe.ps1
# Requires: pip install numpy scipy numba torch tifffile pyinstaller
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..\..")
Set-Location $here

# Only the registration-related suite2p sources are bundled (see mc_core._load_registration).
pyinstaller --noconfirm --clean --windowed --name Suite2pMotionCorrection `
  --add-data "$root\suite2p\default_ops.py;suite2p" `
  --add-data "$root\suite2p\io\binary.py;suite2p\io" `
  --add-data "$root\suite2p\registration\register.py;suite2p\registration" `
  --add-data "$root\suite2p\registration\rigid.py;suite2p\registration" `
  --add-data "$root\suite2p\registration\nonrigid.py;suite2p\registration" `
  --add-data "$root\suite2p\registration\utils.py;suite2p\registration" `
  --add-data "$root\suite2p\registration\bidiphase.py;suite2p\registration" `
  --hidden-import numba --hidden-import torch --hidden-import scipy.ndimage --hidden-import scipy.signal --hidden-import scipy.fft --hidden-import scipy.fftpack --hidden-import tifffile `
  --exclude-module cellpose --exclude-module rastermap --exclude-module matplotlib `
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module sklearn `
  app.py
Write-Host "Output: $here\dist\Suite2pMotionCorrection\Suite2pMotionCorrection.exe"

