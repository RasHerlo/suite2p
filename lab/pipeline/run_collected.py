#!/usr/bin/env python3
"""Collected share-align MC + locked temporal/anatomical extraction.

Walk a ThorImage tree. Inputs are ``ChanA/B_stk_defringed_v22.tif`` in
``DATA/ChanA`` and ``DATA/ChanB`` (defringe repo owns that suffix).

Outputs stay in those same Chan folders so s2p_Trace_Curation can open
both arms and write its pickle beside them::

    DATA/ChanA/
      ChanA_stk_defringed_v22.tif
      MeanImg.png
      suite2p_temp/plane0/   temporal ROIs + F/Fneu
      suite2p_anat/plane0/   anatomical ROIs + F/Fneu
    DATA/ChanB/              same + independent_meanImg.png (fringe guide)

``data.bin`` is written once (share-aligned) under ``suite2p_temp`` and
hardlinked into ``suite2p_anat``. OASIS off. fs / µm/px from Experiment.xml.

    python lab/pipeline/run_collected.py --gui
    python lab/pipeline/run_collected.py --root PATH
    python lab/pipeline/run_collected.py --root PATH --inventory
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tifffile import TiffFile

from lab.configs.defaults import (
    DEFAULT_INPUT_TIFF_TEMPLATE,
    REGISTRATION,
    SEG_EVAL,
    SENSOR_TO_CELL_TYPE,
    acquisition_from_xml,
    apply_microscope_to_registration,
    apply_s2p_ops,
    apply_seg_eval_ops,
    microscope_for_path,
    pmt_roles,
)
from lab.pipeline.compare_mc_channels import compare
from lab.pipeline.fringe_robust_register import find_input_tiff, process_tree
from lab.pipeline.run_seg_eval import plane0_is_complete
from lab.pipeline.run_tau_sweep_level3b_chanb import write_empty_plane
from suite2p import default_ops
from suite2p.run_s2p import run_s2p

LETTERS = ("A", "B")
METHODS = tuple(SEG_EVAL["methods"])
COLLECTED_FOLDERS = SEG_EVAL["collected_folders"]


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def chan_dir(data: Path, letter: str) -> Path:
    return Path(data) / f"Chan{letter}"


def collected_plane(data: Path, letter: str, method: str) -> Path:
    folder = COLLECTED_FOLDERS[method]
    return chan_dir(data, letter) / folder / "plane0"


def tiff_nframes(path: Path) -> int | None:
    if path is None or not Path(path).is_file():
        return None
    with TiffFile(str(path)) as tf:
        if tf.series:
            shape = tf.series[0].shape
            if len(shape) >= 3:
                return int(shape[0])
        return int(len(tf.pages))


def input_name(letter: str, template: str) -> str:
    return template.format(letter=letter, Letter=letter)


def data_root(session: Path) -> Path | None:
    session = Path(session)
    if (session / "DATA" / "ChanA").is_dir() and (session / "DATA" / "ChanB").is_dir():
        return session / "DATA"
    if (session / "ChanA").is_dir() and (session / "ChanB").is_dir():
        return session
    return None


def tiff_for(session: Path, letter: str, template: str) -> Path | None:
    root = data_root(session)
    if root is None:
        return None
    folder = root / f"Chan{letter}"
    cfg = {"input_tiff_template": template, "input_tiff_names": ()}
    return find_input_tiff(folder, cfg, letter=letter)


def is_session(folder: Path, template: str) -> bool:
    folder = Path(folder)
    if not (folder / "Experiment.xml").is_file():
        return False
    return all(tiff_for(folder, L, template) for L in LETTERS)


def discover_sessions(root: Path, template: str) -> list[Path]:
    root = Path(root)
    if is_session(root, template):
        return [root]
    found = []
    for xml in root.rglob("Experiment.xml"):
        session = xml.parent
        if "mc_runs" in session.parts:
            continue
        if is_session(session, template):
            found.append(session)
    return sorted(set(found))


def mapping_from_cells(chan_a: str, chan_b: str) -> dict:
    a, b = str(chan_a).lower(), str(chan_b).lower()
    if {a, b} != {"neuron", "astrocyte"}:
        raise ValueError(
            "ChanA and ChanB must be one neuron and one astrocyte "
            f"(got ChanA={a!r}, ChanB={b!r})"
        )
    return {"A": a, "B": b}


def infer_cell_type(sensor: str, fallback: str) -> str:
    return SENSOR_TO_CELL_TYPE.get(sensor, fallback)


def mc_cfg(template: str, mapping: dict, write_registered_tif: bool = False):
    cfg = deepcopy(REGISTRATION)
    cfg["share_shifts_across_channels"] = True
    cfg["write_registered_tif"] = bool(write_registered_tif)
    cfg["write_data_bin"] = True
    cfg["input_tiff_template"] = template
    cfg["input_tiff_names"] = ()
    cfg["channel_cell_types"] = mapping
    cfg["bin_save_folder"] = COLLECTED_FOLDERS["temporal"]
    return cfg


def inspect_session(session: Path, template: str, mapping: dict,
                    min_nframes: int = 500) -> dict:
    session = Path(session)
    rec = {
        "session": str(session),
        "status": "ok",
        "reason": None,
        "tiffs": {},
        "nframes": {},
        "acquisition": acquisition_from_xml(session),
        "microscope": microscope_for_path(session),
    }
    for letter in LETTERS:
        tif = tiff_for(session, letter, template)
        rec["tiffs"][letter] = str(tif) if tif else None
        rec["nframes"][letter] = tiff_nframes(tif) if tif else None
    if not rec["tiffs"]["A"] or not rec["tiffs"]["B"]:
        rec["status"] = "skip"
        rec["reason"] = f"missing {template} under ChanA/ChanB"
        return rec
    na, nb = rec["nframes"]["A"], rec["nframes"]["B"]
    if na is None or nb is None or na != nb:
        rec["status"] = "skip"
        rec["reason"] = f"unequal nframes A={na} B={nb} (assemble/preview?)"
        return rec
    if min_nframes and min(na, nb) < int(min_nframes):
        rec["status"] = "skip"
        rec["reason"] = f"nframes {na} < min_nframes {min_nframes}"
        return rec
    rec["roles"] = pmt_roles(
        computer=(rec["microscope"] or {}).get("computer"),
        channel_cell_types=mapping,
    )
    rec["cell_type"] = {"A": mapping["A"], "B": mapping["B"]}
    rec["align_channel"] = rec["roles"]["neuron"]
    return rec


def run_mc_session(session: Path, template: str, mapping: dict,
                   overwrite: bool, write_registered_tif: bool = False,
                   min_nframes: int = 500) -> dict:
    info = inspect_session(session, template, mapping, min_nframes=min_nframes)
    if info["status"] != "ok":
        return info
    data = data_root(session)
    cfg = apply_microscope_to_registration(
        mc_cfg(template, mapping, write_registered_tif=write_registered_tif),
        session,
        channel_cell_types=mapping,
    )
    source = {
        "input_data": str(data),
        "input_template": template,
        "tiffs": info["tiffs"],
        "nframes": info["nframes"],
        "acquisition": info["acquisition"],
        "cell_type_by_channel": mapping,
        "align_channel": cfg.get("align_channel"),
        "protocol": "share-align on neuron PMT; independent non-align mean is a fringe guide",
        "suite2p_temp": COLLECTED_FOLDERS["temporal"],
        "suite2p_anat": COLLECTED_FOLDERS["cyto3"],
    }
    (data / "SOURCE.json").write_text(json.dumps(source, indent=2, default=str), encoding="utf-8")
    print(f"\n######## MC {session.name} -> {data} ########")
    process_tree(
        data,
        cfg,
        share_shifts=True,
        overwrite=overwrite,
        output_root=data,
    )
    a_bin = collected_plane(data, "A", "temporal") / "data.bin"
    b_bin = collected_plane(data, "B", "temporal") / "data.bin"
    if not a_bin.exists() or not b_bin.exists():
        info["status"] = "failed"
        info["reason"] = "MC did not write both data.bin files"
        return info
    try:
        compare(data, out_path=data / "compare_AB.png")
    except Exception as exc:
        print(f"  compare_AB skipped: {exc}")
    info["status"] = "ok"
    info["output"] = str(data)
    info["data_bin"] = {"A": str(a_bin), "B": str(b_bin)}
    non_align = "B" if cfg.get("align_channel") == "A" else "A"
    info["shift_warning"] = (chan_dir(data, non_align) / "SHIFT_AGREEMENT_WARNING.txt").exists()
    info["independent_guide"] = str(chan_dir(data, non_align) / "independent_meanImg.png")
    return info


def clone_plane(src_plane: Path, dest_plane: Path) -> Path:
    dest_plane.mkdir(parents=True, exist_ok=True)
    _hardlink_or_copy(src_plane / "data.bin", dest_plane / "data.bin")
    shutil.copy2(src_plane / "ops.npy", dest_plane / "ops.npy")
    for name in ("stat.npy", "F.npy", "Fneu.npy", "iscell.npy", "spks.npy"):
        leftover = dest_plane / name
        if leftover.exists():
            leftover.unlink()
    return dest_plane


def run_seg_arm(data: Path, letter: str, method: str, mapping: dict,
                overwrite: bool) -> Path:
    src = collected_plane(data, letter, "temporal")
    plane = collected_plane(data, letter, method)
    save_folder = COLLECTED_FOLDERS[method]
    chan = chan_dir(data, letter)
    if not (src / "data.bin").exists():
        raise FileNotFoundError(f"missing {src / 'data.bin'}")
    if plane0_is_complete(plane) and not overwrite:
        print(f"  skip complete {plane}")
        return plane
    print(f"\n======== {data.parent.name} Chan{letter} {method} -> {save_folder} ========")
    if plane != src:
        clone_plane(src, plane)
    ops = apply_s2p_ops(default_ops(), tif_path=None, output_dir=chan, start=src)
    ops["channel_cell_types"] = mapping
    apply_seg_eval_ops(
        ops,
        method=method,
        channel_letter=letter,
        computer=ops.get("computer"),
        start=src,
        channel_cell_types=mapping,
    )
    ops["do_registration"] = 0
    ops["roidetect"] = True
    ops["spikedetect"] = False
    ops["save_path0"] = str(chan)
    ops["save_folder"] = save_folder
    ops["fast_disk"] = str(chan)
    try:
        run_s2p(
            ops=ops,
            db={
                "save_path0": str(chan),
                "save_folder": save_folder,
                "fast_disk": str(chan),
                "data_path": [str(src)],
            },
        )
    except ValueError as exc:
        if "no ROIs were found" not in str(exc):
            raise
        print("  detect found 0 ROIs")
        write_empty_plane(plane)
    missing = [n for n in SEG_EVAL["plane0_required"] if not (plane / n).exists()]
    if missing:
        raise FileNotFoundError(f"{plane} missing {missing}")
    print(f"  s2p_Trace_Curation opens {plane.parent}; pickle in {chan}")
    return plane


def write_manifest(data: Path, mapping: dict) -> Path:
    non_align = "B"
    src = data / "SOURCE.json"
    if src.exists():
        align = json.loads(src.read_text(encoding="utf-8")).get("align_channel", "A")
        non_align = "B" if align == "A" else "A"
    man = {
        "data": str(data),
        "cell_type_by_channel": mapping,
        "mc": {
            "A": str(chan_dir(data, "A")),
            "B": str(chan_dir(data, "B")),
            "compare_AB": str(data / "compare_AB.png"),
            "independent_guide": str(chan_dir(data, non_align) / "independent_meanImg.png"),
        },
        "seg": {},
        "trace_curation": {},
        "pickle_dir": {
            "A": str(chan_dir(data, "A")),
            "B": str(chan_dir(data, "B")),
        },
        "pickle_hint": (
            "Open suite2p_temp / suite2p_anat (parent of plane0). "
            "Leave trc_curation.pkl in DATA/ChanA|B beside those folders."
        ),
    }
    for letter in LETTERS:
        for method in METHODS:
            plane = collected_plane(data, letter, method)
            man["seg"][f"{letter}_{method}"] = str(plane)
            man["trace_curation"][f"{letter}_{method}"] = str(plane.parent)
    path = data / "MANIFEST.json"
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return path


def run_seg_session(data: Path, mapping: dict, overwrite: bool) -> dict:
    rec = {"output": str(data), "arms": {}}
    for letter in LETTERS:
        for method in METHODS:
            plane = run_seg_arm(data, letter, method, mapping, overwrite)
            rec["arms"][f"{letter}_{method}"] = str(plane)
    rec["manifest"] = str(write_manifest(data, mapping))
    rec["status"] = "ok"
    return rec


def run_tree(
    root: Path,
    *,
    template: str = DEFAULT_INPUT_TIFF_TEMPLATE,
    chan_a_cell: str = "neuron",
    chan_b_cell: str = "astrocyte",
    chan_a_sensor: str | None = None,
    chan_b_sensor: str | None = None,
    do_mc: bool = True,
    do_seg: bool = True,
    overwrite: bool = False,
    inventory: bool = False,
    write_registered_tif: bool = False,
    min_nframes: int = 500,
) -> list[dict]:
    root = Path(root)
    mapping = mapping_from_cells(chan_a_cell, chan_b_cell)
    sessions = discover_sessions(root, template)
    print(f"Found {len(sessions)} session(s) under {root}")
    summary = []
    for session in sessions:
        data = data_root(session)
        rec = inspect_session(session, template, mapping, min_nframes=min_nframes)
        rec["data"] = str(data) if data else None
        rec["sensors"] = {"A": chan_a_sensor, "B": chan_b_sensor}
        rec["suite2p_temp"] = COLLECTED_FOLDERS["temporal"]
        rec["suite2p_anat"] = COLLECTED_FOLDERS["cyto3"]
        if inventory or rec["status"] != "ok":
            print(f"  {rec['status']:6} {session}  {rec.get('reason') or ''}")
            summary.append(rec)
            continue
        try:
            if do_mc:
                rec = run_mc_session(
                    session, template, mapping, overwrite,
                    write_registered_tif=write_registered_tif,
                    min_nframes=min_nframes,
                )
                rec["data"] = str(data)
            if do_seg:
                temp_bin = collected_plane(data, "A", "temporal") / "data.bin"
                if not temp_bin.exists():
                    rec["status"] = "failed"
                    rec["reason"] = "seg requested but MC data.bin missing"
                else:
                    seg = run_seg_session(data, mapping, overwrite)
                    rec["seg"] = seg
                    rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "failed"
            rec["reason"] = str(exc)
            print(f"  FAILED {session}: {exc}")
        summary.append(rec)
        print(f"  -> {rec.get('status')}  {session.name}")
    summary_path = root / "collected_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {summary_path}")
    except OSError as exc:
        print(f"Could not write summary: {exc}")
    return summary


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Share-align MC then locked temporal + anatomical extraction."
    )
    p.add_argument("--root", default=None, help="Session or parent tree to walk")
    p.add_argument("--gui", action="store_true", help="Open the initiator window")
    p.add_argument("--inventory", action="store_true", help="List sessions; do not run")
    p.add_argument("--skip-mc", action="store_true")
    p.add_argument("--skip-seg", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-stack", action="store_true",
                   help="Also write combined_registered.tif (large)")
    p.add_argument("--chanA-cell", default="neuron", choices=["neuron", "astrocyte"])
    p.add_argument("--chanB-cell", default="astrocyte", choices=["neuron", "astrocyte"])
    p.add_argument("--chanA-sensor", default="jRGECO")
    p.add_argument("--chanB-sensor", default="G-Flamp")
    p.add_argument(
        "--input-template",
        default=DEFAULT_INPUT_TIFF_TEMPLATE,
        help="TIFF name with {letter}, default Chan{letter}_stk_defringed_v22.tif",
    )
    p.add_argument("--min-nframes", type=int, default=500,
                   help="Skip sessions shorter than this (wavelength tests)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gui or args.root is None:
        from lab.pipeline.collected_gui import launch

        return launch(args)
    if not Path(args.root).exists():
        print(f"ERROR: {args.root} does not exist")
        return 1
    run_tree(
        Path(args.root),
        template=args.input_template,
        chan_a_cell=args.chanA_cell,
        chan_b_cell=args.chanB_cell,
        chan_a_sensor=args.chanA_sensor,
        chan_b_sensor=args.chanB_sensor,
        do_mc=not args.skip_mc,
        do_seg=not args.skip_seg,
        overwrite=args.overwrite,
        inventory=args.inventory,
        write_registered_tif=args.save_stack,
        min_nframes=args.min_nframes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
