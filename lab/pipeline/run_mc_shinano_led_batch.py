#!/usr/bin/env python3
"""Share-A cell-ops MC on the five Shinano 260511 LED FOVs (v22).

Writes a sidecar tree (never into original DATA/):

    F:\\bPACNewData2026\\AC_cAMP_Neu_Ca_C1_C2\\mc_runs\\260511\\C1_RLV_LW_maybe\\
        LED_x15_Level{1,3,3b,5_001,5b}\\ChanA|B\\

Each channel gets offsets, registered mean, suite2p/plane0/data.bin
(for later temporal / Cellpose detection), and ChanB also gets the
independent-MC fringe guide.

Skip wavelength tests. Skip 260616 (Musashi, unequal nframes until
assemble fix). Skip 260510\\C1_RW\\Trial (v22 ChanB n=1).

    python lab/pipeline/run_mc_shinano_led_batch.py
    python lab/pipeline/run_mc_shinano_led_batch.py --overwrite
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.configs.defaults import REGISTRATION
from lab.pipeline.compare_mc_channels import compare
from lab.pipeline.fringe_robust_register import process_tree

COLLECTION = Path(r"F:\bPACNewData2026") / "AC_cAMP_Neu_Ca_C1_C2"
ANIMAL = COLLECTION / "260511" / "C1_RLV_LW_maybe"
MC_ROOT = COLLECTION / "mc_runs" / "260511" / "C1_RLV_LW_maybe"
SESSIONS = (
    "LED_x15_Level1",
    "LED_x15_Level3",
    "LED_x15_Level3b",
    "LED_x15_Level5_001",
    "LED_x15_Level5b",
)
SANDBOX = (
    Path(r"F:\bPACNewData2026") / "PreProcessing Optimization" / "Level3b copy"
)
SIG = {
    "A": SANDBOX / "defringe_runs" / "v22_full_seeded500" / "ChanA" / "diagnostics" / "signature.json",
    "B": SANDBOX / "defringe_runs" / "v22_full_seeded500" / "ChanB" / "diagnostics" / "signature.json",
}


def _cfg():
    cfg = deepcopy(REGISTRATION)
    cfg["share_shifts_across_channels"] = True
    cfg["align_channel"] = "A"
    cfg["write_registered_tif"] = False
    cfg["write_data_bin"] = True
    return cfg


def session_ok(session: Path) -> str | None:
    data = session / "DATA"
    a = data / "ChanA" / "ChanA_stk_defringed_v22.tif"
    b = data / "ChanB" / "ChanB_stk_defringed_v22.tif"
    if not a.is_file() or not b.is_file():
        return f"missing v22 stk ({a.exists()=}, {b.exists()=})"
    xml = session / "Experiment.xml"
    if not xml.is_file():
        return "missing Experiment.xml"
    return None


def run_one(name: str, overwrite: bool) -> dict:
    session = ANIMAL / name
    err = session_ok(session)
    out = MC_ROOT / name
    rec = {"session": name, "input": str(session / "DATA"), "output": str(out)}
    if err:
        rec["status"] = "skip"
        rec["reason"] = err
        print(f"\n######## SKIP {name}: {err} ########")
        return rec
    out.mkdir(parents=True, exist_ok=True)
    (out / "SOURCE.txt").write_text(
        f"input DATA: {session / 'DATA'}\n"
        "kind: ChanA/B *_stk_defringed_v22.tif\n"
        "protocol: share-A cell-ops, independent ChanB guide\n"
        "do not write into original DATA/\n",
        encoding="utf-8",
    )
    print(f"\n######## {name} -> {out} ########")
    process_tree(
        session / "DATA",
        _cfg(),
        share_shifts=True,
        overwrite=overwrite,
        output_root=out,
        align_override="A",
    )
    a_off = out / "ChanA" / "offsets.npz"
    b_off = out / "ChanB" / "offsets.npz"
    a_bin = out / "ChanA" / "suite2p" / "plane0" / "data.bin"
    b_bin = out / "ChanB" / "suite2p" / "plane0" / "data.bin"
    if not a_off.exists() or not b_off.exists():
        rec["status"] = "failed"
        rec["reason"] = "missing offsets.npz"
        return rec
    compare(
        out,
        out_path=out / "compare_AB.png",
        signature_a=SIG["A"] if SIG["A"].exists() else None,
        signature_b=SIG["B"] if SIG["B"].exists() else None,
    )
    agree = out / "ChanB" / "shift_agreement.json"
    rec["status"] = "ok"
    rec["data_bin"] = {"A": a_bin.exists(), "B": b_bin.exists()}
    rec["shift_agreement"] = json.loads(agree.read_text(encoding="utf-8")) if agree.exists() else None
    rec["compare"] = str(out / "compare_AB.png")
    warn = out / "ChanB" / "SHIFT_AGREEMENT_WARNING.txt"
    rec["shift_warning"] = warn.exists()
    return rec


def main():
    overwrite = "--overwrite" in sys.argv
    MC_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in SESSIONS:
        rec = run_one(name, overwrite)
        summary.append(rec)
        print(f"  -> {rec['status']}")
    path = MC_ROOT / "batch_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    n_ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"Done: {n_ok}/{len(summary)} sessions ok")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
