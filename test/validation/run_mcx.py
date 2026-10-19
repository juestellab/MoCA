#!/usr/bin/env python3
"""MCX side of the cross-validation, for the phantoms in phantoms.json.

Writes results/mcx_<case>_rep<k>.h5 with the absorbed fraction per voxel per
launched photon (mua * fluence * dV, the quantity Moca returns) and the label
volume. Physics is matched to Moca: n = 1 everywhere, no Fresnel, one CW gate.
Each run checks the conversion against MCX's own energyabs tally.

numpy is C-ordered, so these arrays read back in Julia as (z, y, x).

Run:  uv run python run_mcx.py --all [--quick]
      uv run python run_mcx.py --case C3_semiinf
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import h5py
import numpy as np
import pmcx

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
NPHOTON = 100_000_000      # per replicate; MCX tallies fluence in float32, which loses deposits at 1e9
REPLICATES = 10
SEED = 20260824


def build_volume(case: dict) -> tuple[np.ndarray, list[list[float]]]:
    """Return (labels, prop). Label 0 is MCX's exterior, where photons are
    terminated; must match common.jl::build_volume."""
    nx, ny, nz = case["dim"]
    vol = np.zeros((nx, ny, nz), dtype=np.uint8)

    prop = [[0.0, 0.0, 1.0, 1.0]]  # row 0: exterior
    for i, layer in enumerate(case["layers"], start=1):
        prop.append([layer["mua"], layer["mus"], layer["g"], 1.0])
        vol[:, :, layer["z_from"]:layer["z_to"]] = i

    s = case["shell_voxels"]
    if s > 0:
        vol[:s, :, :] = 0
        vol[-s:, :, :] = 0
        vol[:, :s, :] = 0
        vol[:, -s:, :] = 0
        vol[:, :, :s] = 0
        vol[:, :, -s:] = 0

    return vol, prop


def make_cfg(case: dict, nphoton: int, seed: int) -> dict:
    vol, prop = build_volume(case)
    src = case["source"]
    vx = case["voxel_mm"]
    return {
        "nphoton": int(nphoton),
        "vol": vol,
        "prop": prop,
        "tstart": 0.0,
        "tend": 5e-8,
        "tstep": 5e-8,          # one CW gate
        "srcpos": [p / vx for p in src["pos_mm"]],   # grid units
        "issrcfrom0": 1,        # srcpos is 0-based, as in Moca (pmcx default is 1-based)
        "srcdir": list(src["dir"]),
        "srctype": src["type"],
        "unitinmm": float(vx),
        "isreflect": 0,
        "isspecular": 0,
        "isnormalized": 1,
        "outputtype": "fluence",
        "autopilot": 1,
        "gpuid": 1,
        "seed": int(seed),
    }


def absorbed_fraction(case: dict, cfg: dict, res: dict) -> np.ndarray:
    """Absorbed fraction per voxel per photon: mua * phi * dV."""
    phi = np.asarray(res["flux"])
    phi = phi.reshape(phi.shape[:3] + (-1,)).sum(axis=3)  # collapse time gates

    mua = np.zeros(phi.shape, dtype=np.float64)
    for layer in case["layers"]:
        mua[:, :, layer["z_from"]:layer["z_to"]] = layer["mua"]
    mua[np.asarray(cfg["vol"]) == 0] = 0.0

    return mua * phi * float(cfg["unitinmm"]) ** 3


def run_case(case: dict, quick: bool = False) -> None:
    name = case["name"]
    nphoton = 2_000_000 if quick else NPHOTON
    reps = 1 if quick else REPLICATES
    RESULTS.mkdir(exist_ok=True)
    print(f"\n=== {name}: {case['description']}")

    for rep in range(reps):
        cfg = make_cfg(case, nphoton, SEED + rep)

        t0 = time.time()
        res = pmcx.run(cfg)
        dt = time.time() - t0

        absorbed = absorbed_fraction(case, cfg, res)
        stat = res["stat"]
        closure = absorbed.sum() / (stat["energyabs"] / stat["energytot"]) - 1.0
        print(f"    rep {rep}: {dt:6.1f}s  {nphoton/dt/1e6:6.1f} Mphoton/s  "
              f"absorbed={absorbed.sum():.6f}  closure={closure:+.2e}")
        if abs(closure) > 5e-3:
            print(f"      !! closure {closure:+.2e} against MCX's energyabs exceeds 5e-3 -- "
                  f"the fluence -> absorbed conversion no longer holds", file=sys.stderr)

        with h5py.File(RESULTS / f"mcx_{name}_rep{rep}.h5", "w") as f:
            f.create_dataset("absorbed", data=absorbed.astype(np.float64))
            f.create_dataset("label", data=np.asarray(cfg["vol"]))
            f.attrs["nphoton"] = nphoton
            f.attrs["pmcx_version"] = pmcx.__version__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--quick", action="store_true", help="2e6 photons, 1 replicate")
    args = ap.parse_args()

    cases = json.loads((HERE / "phantoms.json").read_text())["cases"]
    if not args.all:
        if not args.case:
            ap.error("pass --all or --case NAME")
        cases = [c for c in cases if c["name"] in set(args.case)]
        if not cases:
            ap.error(f"no case matched {args.case}")

    for c in cases:
        run_case(c, quick=args.quick)


if __name__ == "__main__":
    main()