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
    "spatial_scale": 0,  # neuron default (auto). Astrocytes: TEMPORAL_BY_CELL_TYPE (µm → scale).
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
# ChanA/B are PMT paths, not cell types.
# MICROSCOPES = cube defaults for the *current* sensors (G-Flamp green,
# jRGECO/RCaMP red). If those fluorophores move, set CHANNEL_CELL_TYPES
# (and align to the neuron PMT). Temporal/Cellpose ops follow cell type.
MICROSCOPES = {
    "THORLABS_30_016": {
        "name": "Shinano",
        "align_channel": "A",
        "neuron": "A",
        "astrocyte": "B",
    },
    "USER-PC": {
        "name": "Musashi",
        "align_channel": "B",
        "neuron": "B",
        "astrocyte": "A",
    },
}


# Prep-level PMT → cell type. Empty = use MICROSCOPES cube defaults.
# Override when sensors change, e.g. {"A": "astrocyte", "B": "neuron"}.
CHANNEL_CELL_TYPES = {}

# Fluorophore → default cell type. GUI sensors fill cell-type boxes; ops
# still follow the cell-type boxes (astrocytes / neurons), not Chan letters.
SENSORS = ("G-Flamp", "jRGECO", "RCaMP", "other")
SENSOR_TO_CELL_TYPE = {
    "G-Flamp": "astrocyte",
    "jRGECO": "neuron",
    "RCaMP": "neuron",
}

DEFAULT_INPUT_TIFF_TEMPLATE = "Chan{letter}_stk_defringed_v22.tif"


def find_experiment_xml(start):
    """Walk up a few parents and one child level for ThorImage Experiment.xml.

    MC sidecars live under ``mc_runs/``. Those folders do not contain the
    XML; strip that segment and look in the original session tree.
    """
    start = Path(start)
    if start.is_file():
        start = start.parent
    search = [start, *list(start.parents)[:6]]
    parts = list(start.parts)
    if "mc_runs" in parts:
        i = parts.index("mc_runs")
        alt = Path(*parts[:i]) / Path(*parts[i + 1 :])
        search.extend([alt, *list(alt.parents)[:5]])
    seen = set()
    for folder in search:
        folder = Path(folder)
        key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        xml = folder / "Experiment.xml"
        if xml.is_file():
            return xml
    for xml in start.glob("Experiment.xml"):
        return xml
    for xml in start.glob("*/Experiment.xml"):
        return xml
    return None


def computer_name_from_xml(xml_path):
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag.lower() == "computer":
            return el.attrib.get("name") or el.attrib.get("Name")
    return None


def _xml_elem(xml_path, name):
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    want = name.lower()
    for el in root.iter():
        if el.tag.split("}")[-1].lower() == want:
            return el
    return None


def acquisition_from_xml(start):
    """ThorImage LSM pixel size and effective frame rate from Experiment.xml.

    ``fs`` = ``frameRate / averageNum`` (line averaging). ``um_per_px`` is
    ``LSM/@pixelSizeUM``, else ``widthUM / pixelX``.
    """
    xml = find_experiment_xml(start) if start is not None else None
    if xml is None:
        return None
    lsm = _xml_elem(xml, "LSM")
    attrib = lsm.attrib if lsm is not None else {}
    frame_rate = float(attrib.get("frameRate") or 0)
    average_num = float(attrib.get("averageNum") or 1) or 1.0
    pixel_x = float(attrib.get("pixelX") or 0)
    width_um = float(attrib.get("widthUM") or 0)
    um_per_px = float(attrib.get("pixelSizeUM") or 0)
    if um_per_px <= 0 and pixel_x > 0 and width_um > 0:
        um_per_px = width_um / pixel_x
    fs = frame_rate / average_num if frame_rate > 0 else None
    return {
        "xml": str(xml),
        "computer": computer_name_from_xml(xml),
        "frame_rate": frame_rate,
        "average_num": average_num,
        "fs": fs,
        "um_per_px": um_per_px if um_per_px > 0 else None,
        "pixel_x": pixel_x,
        "width_um": width_um,
    }


def apply_acquisition_to_ops(ops, start, computer=None):
    """Set ``fs`` and ``um_per_px`` from Experiment.xml. Does not detect ROIs."""
    acq = acquisition_from_xml(start) if start is not None else None
    if acq is None:
        print("  WARNING: no Experiment.xml near path; fs / µm/px not updated")
        return ops
    if acq.get("fs"):
        ops["fs"] = float(acq["fs"])
    if acq.get("um_per_px"):
        ops["um_per_px"] = float(acq["um_per_px"])
    ops["experiment_xml"] = acq["xml"]
    ops["frame_rate_xml"] = acq["frame_rate"]
    ops["average_num"] = acq["average_num"]
    if acq.get("computer"):
        ops["computer"] = acq["computer"]
    elif computer:
        ops["computer"] = computer
    print(
        f"  XML {Path(acq['xml']).name}: fs={ops.get('fs')} Hz  "
        f"um/px={ops.get('um_per_px')}"
    )
    return ops


def microscope_for_path(start):
    """Map a session folder to MICROSCOPES via Experiment.xml Computer name."""
    xml = find_experiment_xml(start)
    if xml is None:
        return None
    computer = computer_name_from_xml(xml)
    if not computer:
        return None
    info = MICROSCOPES.get(computer)
    if info is None:
        return {
            "name": "unknown",
            "computer": computer,
            "align_channel": None,
            "xml": str(xml),
        }
    return {**info, "computer": computer, "xml": str(xml)}


def apply_microscope_to_registration(cfg, start, align_override=None,
                                    channel_cell_types=None):
    """Set align_channel to the *neuron* PMT. Never mutates cfg.

    Cube defaults come from Experiment.xml Computer name. Per-run
    channel_cell_types (GUI / CLI) wins, then CHANNEL_CELL_TYPES, then
    MICROSCOPES. A fluorophore swap moves share-align with the neuron PMT.
    """
    cfg = deepcopy(cfg)
    mapping = (
        channel_cell_types
        if channel_cell_types is not None
        else cfg.get("channel_cell_types") or CHANNEL_CELL_TYPES
    )
    if align_override:
        cfg["align_channel"] = str(align_override).upper()
        cfg["microscope_source"] = "cli"
        if mapping:
            cfg["channel_cell_types"] = mapping
        return cfg
    info = microscope_for_path(start)
    if info is not None:
        cfg["computer"] = info.get("computer")
        cfg["microscope"] = info.get("name")
        cfg["microscope_source"] = info.get("xml")
    if not mapping and info is None:
        return cfg
    roles = pmt_roles(computer=cfg.get("computer"), channel_cell_types=mapping)
    cfg["align_channel"] = roles["neuron"]
    cfg["channel_cell_types"] = {
        roles["neuron"]: "neuron",
        roles["astrocyte"]: "astrocyte",
    }
    src = "GUI/CLI cell types" if mapping else f"microscope {cfg.get('microscope')}"
    print(f"  {src}: neuron=Chan{roles['neuron']} astro=Chan{roles['astrocyte']} "
          f"align_channel={roles['neuron']}")
    return cfg

# Default: estimate align_channel, apply those shifts to the other PMT.
# Also estimate the non-align PMT independently: keep that meanImg as an
# ROI-curation guide (residual fringes) and warn if its shifts disagree.
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
    "share_shift_warn_pearson": 0.7,
    "share_shift_warn_median_px": 2.0,
    "write_registered_tif": True,
    "write_data_bin": False,
    # Prefer this template (letter A/B). Defringe repo owns the suffix.
    "input_tiff_template": DEFAULT_INPUT_TIFF_TEMPLATE,
    "input_tiff_names": (
        "denoised_cut.tif",
        "ChanA_stk.tif",
        "ChanB_stk.tif",
        "ChanA_stk_defringed_v21.tif",
        "ChanB_stk_defringed_v21.tif",
        "ChanA_stk_defringed_v22.tif",
        "ChanB_stk_defringed_v22.tif",
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

# anatomical_only=0 keeps functional detection as the apply_s2p_ops default.
# The locked Cellpose arm is ANATOMICAL_BY_CELL_TYPE (apply_seg_eval_ops
# method "cyto3"), not this dict. Custom weights still go in models.*.
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
        # Shinano default. Musashi (USER-PC): swap A/B (astro on A, neuron on B).
        "ChanA": "neuron",
        "ChanB": "astrocyte",
        "SUPPORT_ChanA": "neuron",
        "SUPPORT_ChanB": "astrocyte",
    },
}


# Locked on Shinano LED_x15 Level3b (LSM pixelSizeUM=0.935, fs=14.80 Hz).
# Temporal and anatomical knobs are *physical*; apply_* converts to pixels
# / suite2p spatial_scale from Experiment.xml um_per_px and fs.
REFERENCE_UM_PER_PX = 0.935
# suite2p sparse scale s → 3 * 2**s pixels (s=1..4 → 6, 12, 24, 48).
SPATIAL_SCALE_PX = {1: 6, 2: 12, 3: 24, 4: 48}

# Temporal / sparse-mode detection. Locked 2026-08-20 from Level3b AC sweeps.
# Follows *astrocytes*, not ChanA/B. PMT letter: CHANNEL_CELL_TYPES or MICROSCOPES.
TEMPORAL_BY_CELL_TYPE = {
    "astrocyte": {
        "tau": 1.0,
        "spatscale_um": 6 * REFERENCE_UM_PER_PX,  # scale 1 at the lock FOV
        "spatial_scale": 1,
    },
    "neuron": {
        "tau": 1.0,
        "spatscale_um": None,  # auto
        "spatial_scale": 0,
    },
}

# Anatomical / Cellpose-on-mean. Locked 2026-08-20 from share-A ChanB cyto3
# diameter 9 / flow 0.4. Intercalate with temporal in s2p_Trace_Curation.
ANATOMICAL_BY_CELL_TYPE = {
    "astrocyte": {
        "anatomical_only": 2,
        "pretrained_model": "cyto3",
        "diameter_um": 9 * REFERENCE_UM_PER_PX,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "spatial_hp_cp": 0,
    },
    "neuron": {
        "anatomical_only": 2,
        "pretrained_model": "cyto3",
        "diameter_um": None,  # Cellpose auto until a neuron diameter is locked
        "flow_threshold": 1.5,
        "cellprob_threshold": 0.0,
        "spatial_hp_cp": 0,
    },
}


def channel_letter_from_name(token):
    text = str(token)
    if "ChanB" in text or text in ("B", "b"):
        return "B"
    if "ChanA" in text or text in ("A", "a"):
        return "A"
    return None


def pmt_roles(computer=None, channel_cell_types=None):
    """Which PMT letter is neuron vs astrocyte for this prep.

    CHANNEL_CELL_TYPES (sensor assignment) wins over MICROSCOPES (cube default).
    """
    mapping = (
        channel_cell_types if channel_cell_types is not None else CHANNEL_CELL_TYPES
    )
    if mapping:
        roles = {}
        for key, cell_type in mapping.items():
            letter = channel_letter_from_name(key) or str(key).upper()
            if cell_type in ("neuron", "astrocyte") and letter in ("A", "B"):
                roles[cell_type] = letter
        if "neuron" in roles and "astrocyte" in roles:
            return roles
    info = MICROSCOPES.get(computer) if computer else None
    if info is None:
        info = MICROSCOPES["THORLABS_30_016"]
    return {"neuron": info["neuron"], "astrocyte": info["astrocyte"]}


def cell_type_for_letter(letter, computer=None, channel_cell_types=None):
    """Map PMT letter to neuron/astrocyte. Follows ACs, not a fixed Chan letter."""
    letter = channel_letter_from_name(letter) or (
        str(letter).upper() if letter else None
    )
    roles = pmt_roles(computer=computer, channel_cell_types=channel_cell_types)
    if letter == roles.get("astrocyte"):
        return "astrocyte"
    if letter == roles.get("neuron"):
        return "neuron"
    return None


def spatial_scale_from_um(spatscale_um, um_per_px):
    """Nearest suite2p sparse scale (1–4) for a physical correlation window."""
    if not spatscale_um or not um_per_px or um_per_px <= 0:
        return None
    best, best_err = 1, float("inf")
    for scale, px in SPATIAL_SCALE_PX.items():
        err = abs(px * um_per_px - spatscale_um)
        if err < best_err:
            best, best_err = scale, err
    return best


def diameter_px_from_um(diameter_um, um_per_px):
    """Cellpose diameter in pixels, or 0 (auto) if either side is missing."""
    if not diameter_um or not um_per_px or um_per_px <= 0:
        return 0
    return max(1, int(round(float(diameter_um) / float(um_per_px))))


def apply_temporal_cell_ops(ops, cell_type, um_per_px=None):
    cfg = TEMPORAL_BY_CELL_TYPE.get(cell_type)
    if not cfg:
        return ops
    ops["tau"] = cfg["tau"]
    ops["cell_type"] = cell_type
    um = um_per_px if um_per_px is not None else ops.get("um_per_px")
    spat_um = cfg.get("spatscale_um")
    if spat_um and um:
        scale = spatial_scale_from_um(spat_um, um)
        ops["spatial_scale"] = int(scale)
        ops["spatscale_um"] = float(spat_um)
        px = SPATIAL_SCALE_PX[scale]
        print(
            f"  temporal {cell_type}: tau={cfg['tau']} s  "
            f"{spat_um:.2f} µm → spatial_scale={scale} ({px} px @ {um:.3f} µm/px)"
        )
    else:
        ops["spatial_scale"] = cfg["spatial_scale"]
        if spat_um and not um:
            print(
                f"  WARNING: no µm/px; temporal {cell_type} spatial_scale="
                f"{cfg['spatial_scale']} (lock-FOV fallback)"
            )
    return ops


def apply_anatomical_cell_ops(ops, cell_type, um_per_px=None):
    cfg = ANATOMICAL_BY_CELL_TYPE.get(cell_type)
    if not cfg:
        return ops
    ops["anatomical_only"] = cfg["anatomical_only"]
    ops["pretrained_model"] = cfg["pretrained_model"]
    ops["flow_threshold"] = cfg["flow_threshold"]
    ops["cellprob_threshold"] = cfg["cellprob_threshold"]
    ops["spatial_hp_cp"] = cfg["spatial_hp_cp"]
    ops["cell_type"] = cell_type
    um = um_per_px if um_per_px is not None else ops.get("um_per_px")
    diam_um = cfg.get("diameter_um")
    if diam_um and um:
        ops["diameter"] = diameter_px_from_um(diam_um, um)
        ops["diameter_um"] = float(diam_um)
        print(
            f"  anatomical {cell_type}: {diam_um:.2f} µm → diameter="
            f"{ops['diameter']} px @ {um:.3f} µm/px  "
            f"flow={cfg['flow_threshold']}  model={cfg['pretrained_model']}"
        )
    else:
        ops["diameter"] = 0
        if diam_um and not um:
            print(
                f"  WARNING: no µm/px; anatomical {cell_type} diameter auto "
                f"(wanted {diam_um:.2f} µm)"
            )
        else:
            print(
                f"  anatomical {cell_type}: diameter=auto  "
                f"flow={cfg['flow_threshold']}  model={cfg['pretrained_model']}"
            )
    custom = CELLPOSE["models"].get(cell_type)
    if custom:
        ops["pretrained_model"] = custom
    return ops


def apply_s2p_ops(ops, tif_path=None, output_dir=None, cell_type=None, computer=None,
                  start=None):
    """Write lab defaults onto a suite2p ops dict.

    Sets fs / µm/px from Experiment.xml when a nearby path is given.
    Applies locked *temporal* cell-type ops. Anatomical (Cellpose) ops are
    applied by apply_seg_eval_ops(method='cyto3'), not here, so a default
    apply_s2p_ops path stays functional detection.
    """
    ops.update(deepcopy(S2P_OPS))

    if tif_path is not None:
        ops["data_path"] = [str(Path(tif_path).parent)]
        start = start or tif_path
        if cell_type is None:
            cell_type = cell_type_for_letter(
                Path(tif_path).parent.name, computer=computer
            )
    if output_dir is not None:
        ops["save_path0"] = str(output_dir)
        ops["save_folder"] = "suite2p"
        ops["fast_disk"] = str(output_dir)
        start = start or output_dir

    apply_acquisition_to_ops(ops, start, computer=computer)
    if cell_type is None and tif_path is not None:
        cell_type = cell_type_for_letter(
            Path(tif_path).parent.name, computer=ops.get("computer")
        )
    apply_temporal_cell_ops(ops, cell_type, um_per_px=ops.get("um_per_px"))
    return ops


# Segmentation comparison runs must be openable in:
#   1) this repo's suite2p GUI (File → load plane0/stat.npy)
#   2) s2p_Trace_Curation (File → Open suite2p folder… = parent of plane0)
# Deconvolution is not part of the eval; suite2p still writes zeros to spks.npy
# so the GUI loader does not reject the folder.
# Default pipeline: run *both* methods on the same registered data.bin;
# intercalation of the two plane0s is s2p_Trace_Curation.
SEG_EVAL = {
    "roidetect": True,
    "neuropil_extract": True,
    "spikedetect": False,
    "delete_bin": False,
    "do_regmetrics": False,
    "save_folder": "suite2p",
    # Collected pipeline (next to defringed_v22 in DATA/ChanA|B):
    "collected_folders": {
        "temporal": "suite2p_temp",
        "cyto3": "suite2p_anat",
    },
    "plane0_required": (
        "ops.npy",
        "stat.npy",
        "F.npy",
        "Fneu.npy",
        "iscell.npy",
        "spks.npy",
        "data.bin",
    ),
    "methods": {
        "temporal": {"anatomical_only": 0},
        "cyto3": {"anatomical_only": 2, "pretrained_model": "cyto3"},
    },
}


def apply_seg_eval_ops(
    ops, method="temporal", cell_type=None, channel_letter=None, computer=None,
    start=None, channel_cell_types=None,
):
    """Detection + F/Fneu extraction, no OASIS. Cell-ops registration overlay.

    ``temporal`` and ``cyto3`` both follow cell type (astrocyte vs neuron) and
    convert locked µm / s using ops['um_per_px'] / ops['fs'] (from XML).
    """
    ops.update({
        "roidetect": True,
        "neuropil_extract": True,
        "spikedetect": False,
        "delete_bin": False,
        "do_regmetrics": False,
        "1Preg": False,
        "smooth_sigma": REGISTRATION["smooth_sigma"],
        "nonrigid": REGISTRATION["nonrigid"],
        "maxregshift": REGISTRATION["maxregshift"],
        "maxregshiftNR": REGISTRATION["maxregshiftNR"],
    })
    method_ops = SEG_EVAL["methods"].get(method)
    if method_ops is None:
        raise ValueError(
            f"unknown SEG_EVAL method {method!r}; "
            f"expected one of {tuple(SEG_EVAL['methods'])}"
        )
    ops.update(method_ops)
    if start is not None:
        apply_acquisition_to_ops(ops, start, computer=computer)
    computer = computer or ops.get("computer")
    if cell_type is None and channel_letter is not None:
        cell_type = cell_type_for_letter(
            channel_letter,
            computer=computer,
            channel_cell_types=channel_cell_types or ops.get("channel_cell_types"),
        )
    if method == "temporal":
        apply_temporal_cell_ops(ops, cell_type, um_per_px=ops.get("um_per_px"))
    elif method == "cyto3":
        apply_anatomical_cell_ops(ops, cell_type, um_per_px=ops.get("um_per_px"))
    return ops
