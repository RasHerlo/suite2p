"""Shared lab defaults.

These values used to be copied across walkers and the master pipeline.
Experiment-specific YAML (see experiment.example.yaml) should override this
module later instead of forking the runner scripts.

Cellpose slots are present so neuron / astrocyte models can be wired in
without changing the current functional-detection path (anatomical_only=0).
"""

from copy import deepcopy
from pathlib import Path


# Applied on top of suite2p.default_ops()
S2P_OPS = {
    "nplanes": 1,
    "nchannels": 1,
    "functional_chan": 1,
    "tau": 1.0,
    "fs": 10.0,
    "do_registration": True,
    "nonrigid": True,
    "block_size": [128, 128],
    "maxregshift": 0.1,
    "align_by_chan": 1,
    "roidetect": True,
    "spikedetect": True,
    "spatial_scale": 0,
    "connected": True,
    "max_overlap": 0.75,
    "neuropil_extract": True,
    "inner_neuropil_radius": 2,
    "min_neuropil_pixels": 350,
    "neucoeff": 0.7,
}

ROI_SELECTION = {
    "ellipticity_threshold": 0.78,
    "components_threshold": 3,
}

# Cell-oriented motion correction (lab/pipeline/fringe_robust_register.py).
# Incoming stacks may already be defringed; this repo does not defringe.
# Do not turn 1Preg on: spatial_hp_reg keeps ~10 px residual fringes.
# ChanA/B are PMT paths; align_channel is a default for Shinano+C1 (red neurons).
REGISTRATION = {
    "output_folder": "suite2p_cellreg",
    "align_filter": "none",  # none | lowpass (lowpass = phasecorr weighting only)
    "lowpass_sigma": 4.0,
    "smooth_sigma": 3.0,
    "nonrigid": False,
    "maxregshift": 0.1,
    "maxregshiftNR": 3,
    "one_p_reg": False,
    "corrxy_frac": 0.4,
    "share_shifts_across_channels": True,
    "align_channel": "A",
    "write_registered_tif": True,
    "input_tiff_names": (
        "denoised_cut.tif",
        "ChanA_stk.tif",
        "ChanB_stk.tif",
        "ChanA_stk_defringed_v21.tif",
        "ChanB_stk_defringed_v21.tif",
        "ChanA_stk_defringed.tif",
        "ChanB_stk_defringed.tif",
    ),
}

# Default MouseLand-like rigid+nonrigid, for bakeoff against REGISTRATION.
REGISTRATION_LEGACY = {
    **REGISTRATION,
    "align_filter": "none",
    "smooth_sigma": 1.15,
    "nonrigid": True,
    "maxregshiftNR": 5,
    "corrxy_frac": 0.0,  # do not interpolate; match stock suite2p
    "share_shifts_across_channels": False,
    "write_registered_tif": False,
}

CHANNEL_FILES = {
    "ChanA": "ChanA_stk.tif",
    "ChanB": "ChanB_stk.tif",
    "SUPPORT_ChanA": "denoised_cut.tif",
    "SUPPORT_ChanB": "denoised_cut.tif",
}

# FFT deripple presets keyed by channel letter (as used by data_processing_master)
FFT_MASKS = {
    "A": {"type": "circle", "coords": [(15, 25)]},
    "B": {
        "type": "rect",
        "coords": [[-52, -81, 43, 7], [52, 81, 43, 7]],
    },
}

# n_frames in F.npy → (start, end) stim window. Replace with per-experiment config later.
STIMULATION_RANGES_BY_NFRAMES = {
    1520: (726, 733),
    2890: (1381, 1388),
}

# anatomical_only=0 keeps current suite2p functional detection.
# Set anatomical_only to 1–4 and point models.* at Cellpose weights to switch.
CELLPOSE = {
    "anatomical_only": 0,
    "diameter": 0,
    "cellprob_threshold": 0.0,
    "flow_threshold": 1.5,
    "pretrained_model": "cyto",
    "models": {
        "neuron": None,
        "astrocyte": None,
    },
    "channel_models": {
        "ChanA": "neuron",
        "ChanB": "astrocyte",
        "SUPPORT_ChanA": "neuron",
        "SUPPORT_ChanB": "astrocyte",
    },
}


def apply_s2p_ops(ops, tif_path=None, output_dir=None):
    """Write lab defaults onto a suite2p ops dict.

    Path fields are set when *tif_path* / *output_dir* are given.
    Cellpose ops are only copied when anatomical_only > 0, so the current
    functional-detection runs stay unchanged.
    """
    ops.update(deepcopy(S2P_OPS))

    if tif_path is not None:
        ops["data_path"] = [str(Path(tif_path).parent)]
    if output_dir is not None:
        ops["save_path0"] = str(output_dir)
        ops["save_folder"] = "suite2p"
        ops["fast_disk"] = str(output_dir)

    if CELLPOSE.get("anatomical_only", 0):
        ops["anatomical_only"] = CELLPOSE["anatomical_only"]
        ops["diameter"] = CELLPOSE["diameter"]
        ops["cellprob_threshold"] = CELLPOSE["cellprob_threshold"]
        ops["flow_threshold"] = CELLPOSE["flow_threshold"]
        cell_type = CELLPOSE["channel_models"].get(Path(tif_path).parent.name) if tif_path else None
        model = CELLPOSE["models"].get(cell_type) if cell_type else None
        ops["pretrained_model"] = model or CELLPOSE["pretrained_model"]

    return ops
