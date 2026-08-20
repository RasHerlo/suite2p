# Handoff: suite2p repo → Thorlabs_Data_Overview

**Audience:** an agent or person working in
[Thorlabs_Data_Overview](https://github.com/RasHerlo/Thorlabs_Data_Overview)
(assemble per-frame Thor tiffs into `ChanA_stk.tif` / `ChanB_stk.tif`).

**This file is the outbound note.** Authoritative copy:
`C:\Users\rasmu\Projects\Repos\suite2p\lab\notes\HANDOFF_FOR_THORLABS_DATA_OVERVIEW.md`

Mirror: `C:\Users\rasmu\Projects\Repos\Thorlabs_Data_Overview\notes\INCOMING_FROM_SUITE2P.md`

Do not motion-correct or defringe here. Suite2p only consumes the assembled
stks (then v22 defringe, then MC).

Last updated: 2026-08-20.

---

## What suite2p needs from you

1. **One frame in the stk = one acquired timepoint.** Do **not** append
   `ChanA_Preview.tif` / `ChanB_Preview.tif` (or any other still) into
   `Chan*_stk.tif`.
2. **ChanA and ChanB stks must have the same `n_frames`.** Share-shift MC
   will refuse unequal lengths.
3. Leave original session `DATA/` layout; we will not write MC into it.

---

## Bug seen on 260616 (Musashi / `USER-PC`)

All six imaging `DATA` folders have **raw ChanA = raw ChanB + 1 frame**.
That extra frame is the preview included in the ChanA stack.

| session | raw A | raw B |
|---|---|---|
| `C1_LRV\10mRunwAPsMid` | 3084 | 3083 |
| `C1_LRV\Run8minwAPsMid_Calib` | 2469 | 2468 |
| `C2_LW\VascEndFeet` | 2469 | 2468 |
| `C2_RV\Run8minwAPsMid_Calib_001` | 2469 | 2468 |
| `C2_RW\Run8minwAPsMid` | 2469 | 2468 |
| `C2_RW\Run8minwAPsMid_000` | 1446 | 1445 |

Root: `F:\bPACNewData2026\AC_cAMP_Neu_Ca_C1_C2\260616\`

v22 defringe then either skipped ChanA or copied the mismatch. After you
rebuild stks **without** the preview, defringe can re-run ChanA (and
`VascEndFeet`, which has no v22 at all).

Also: `260510\C1_RW\Trial` v22 ChanB is 1 frame vs ChanA 360 — likely a
different assemble/defringe miss, not this preview bug.

---

## Microscope (for later MC; not your job to swap tiffs)

`Experiment.xml` `<Computer name="...">`:

| Name | Rig | Green / G-Flamp astro | Red / jRGECO-RCaMP neuron |
|---|---|---|---|
| `THORLABS_30_016` | Shinano | ChanB | ChanA |
| `USER-PC` | Musashi | ChanA | ChanB |

Cubes are inverted on Musashi. Do **not** rename ChanA/B folders to “fix”
that. If `Experiment.xml` is present, suite2p maps `<Computer name>` →
neuron PMT (`align_channel` A on Shinano, B on Musashi).

---

## When a stk is “done”

`DATA/ChanA/ChanA_stk.tif` and `DATA/ChanB/ChanB_stk.tif` exist, same
length, same `Ly`×`Lx`, no preview frames. Then defringe (v22) then
suite2p MC can proceed.
