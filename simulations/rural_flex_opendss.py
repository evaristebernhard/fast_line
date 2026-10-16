"""Fixed-topology IEEE-123 rural-flex equipment screening.

Runs on the existing fast_line IEEE-123 OpenDSS files. The EC-SAV dynamic
solver is not used: this experiment is a quasi-static snapshot comparison.
Assumptions: coordinated tap setting, a new 0/150/300 kvar capacitor bank
at bus 83, and an optional dispatchable 500 kVA PV inverter at bus 83.
The three snapshots are scenario assumptions, NOT a historical rural dataset.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import dss
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ieee123_opendss"
SCENARIOS = [("light_load_high_pv", 0.75, 350.0),
             ("base_load_medium_pv", 1.00, 200.0),
             ("heavy_load_low_pv", 1.25, 80.0)]
V_MIN, V_MAX = 0.95, 1.05


def simulate(tap_offset: int, capacitor_kvar: int, inverter_kvar: int,
             scenario: tuple[str, float, float], data_dir: Path) -> dict:
    name, load_factor, pv_kw = scenario
    if math.hypot(pv_kw, inverter_kvar) > 500.0:
        raise ValueError("PV inverter 500 kVA power circle exceeded")

    ctx = dss.DSS.NewContext().to_opendssdirect()
    ctx.Basic.ClearAll()
    ctx.Text.Command(f"Compile [{data_dir / 'IEEE123Master.dss'}]")
    ctx.Text.Command("Set ControlMode=OFF")
    for load_name in ctx.Loads.AllNames():
        ctx.Loads.Name(load_name)
        kw = float(ctx.Loads.kW())
        kvar = float(ctx.Loads.kvar())
        ctx.Loads.kW(kw * load_factor)
        ctx.Loads.kvar(kvar * load_factor)

    # reg1a is the three-phase source-end regulator in the supplied IEEE-123 data.
    # Control is deliberately frozen to evaluate discrete equipment plans.
    tap = 1.0 + tap_offset * 0.00625
    ctx.Text.Command(f"Edit Transformer.reg1a wdg=2 tap={tap:.6f}")
    if capacitor_kvar:
        ctx.Text.Command("New Capacitor.RuralStep Bus1=83 Phases=3 kV=4.16 "
                         f"kvar={capacitor_kvar}")
    ctx.Text.Command("New Generator.RuralPV Bus1=83 Phases=3 kV=4.16 "
                     f"kW={pv_kw} kvar={inverter_kvar} Model=1")
    ctx.Text.Command("Set Mode=Snapshot")
    ctx.Solution.Solve()

    if not ctx.Solution.Converged():
        raise RuntimeError(f"OpenDSS failed to converge: {name}")
    vm = np.asarray(ctx.Circuit.AllBusMagPu(), dtype=float)
    vm = vm[np.isfinite(vm) & (vm > 0.5) & (vm < 1.5)]
    if not vm.size:
        raise RuntimeError("OpenDSS returned no valid node voltages")
    losses = ctx.Circuit.Losses()
    return {"scenario": name, "load_factor": load_factor, "pv_kw": pv_kw,
            "tap_offset": tap_offset, "capacitor_kvar": capacitor_kvar,
            "inverter_kvar": inverter_kvar, "min_voltage_pu": float(vm.min()),
            "max_voltage_pu": float(vm.max()),
            "voltage_margin_pu": float(min(vm.min() - V_MIN, V_MAX - vm.max())),
            "losses_kw": float(losses[0]) / 1000.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("output/rural_flex"))
    args = parser.parse_args()
    # OpenDSS Compile may change cwd; anchor result paths before compiling.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()

    rows, schemes = [], []
    arms = {"B0_fixed": [(0, 0, 0)],
            "B1_tap_cap": list(itertools.product(range(-2, 3), (0, 150, 300), (0,))),
            "B2_tap_cap_pv": list(itertools.product(range(-2, 3), (0, 150, 300), (-150, 0, 150)))}
    for arm, choices in arms.items():
        for tap, cap, q in choices:
            subset = []
            for s in SCENARIOS:
                result = simulate(tap, cap, q, s, data_dir)
                result["arm"] = arm
                rows.append(result)
                subset.append(result)
            worst = min(x["voltage_margin_pu"] for x in subset)
            avg_loss = float(np.mean([x["losses_kw"] for x in subset]))
            schemes.append({"arm": arm, "tap_offset": tap,
                            "capacitor_kvar": cap, "inverter_kvar": q,
                            "worst_voltage_margin_pu": worst,
                            "mean_losses_kw": avg_loss,
                            "all_snapshots_voltage_safe": bool(worst >= -1e-8),
                            "min_voltage_pu": min(x["min_voltage_pu"] for x in subset),
                            "max_voltage_pu": max(x["max_voltage_pu"] for x in subset)})
    winners = {}
    for arm in arms:
        eligible = [x for x in schemes if x["arm"] == arm]
        winners[arm] = max(eligible,
                           key=lambda x: (x["worst_voltage_margin_pu"],
                                          -x["mean_losses_kw"]))
    with (args.out / "all_scenarios.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "all_configurations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schemes[0].keys())
        writer.writeheader()
        writer.writerows(schemes)
    with (args.out / "winners.json").open("w") as f:
        json.dump(winners, f, indent=2, ensure_ascii=False)
    print(json.dumps(winners, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
