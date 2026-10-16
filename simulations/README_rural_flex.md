# Rural feeder flexible-device screening (IEEE-123)

This is a **steady-state OpenDSS experiment** on the existing IEEE-123 feeder data in `data/ieee123_opendss/`. It does **not** modify or use the EC-SAV dynamic solver.

## Equipment and assumptions

- Keep the IEEE-123 network topology and its original fixed equipment intact.
- Sweep source-end regulator `Transformer.reg1a` winding-2 tap offset `k∈{-2,-1,0,1,2}`, with ratio `1+0.00625k`.
- Add an *assumed* three-phase capacitor bank at bus 83, `C∈{0,150,300}` kvar (this is not an original switched capacitor in the benchmark).
- Add an *assumed* 500 kVA controllable PV generator at bus 83, with reactive dispatch `Q∈{-150,0,150}` kvar. Ensure `sqrt(P²+Q²) ≤ 500 kVA`.
- Three snapshots: load multipliers and PV injection (0.75,350 kW), (1.0,200 kW), (1.25,80 kW).
- Regulator automatic control disabled so candidate tap settings are not overwritten by OpenDSS control actions.
- The ranking criterion is the **minimum voltage margin** over all nodes and all three snapshots relative to [0.95,1.05] pu; ties prefer lower average active loss.

## Reproduce

```
python -m pip install numpy opendssdirect.py 'dss-python>=0.15,<0.16'
python simulations/rural_flex_opendss.py --out output/rural_flex
```

The GitHub Actions workflow `.github/workflows/rural-flex-sim.yml` runs the same experiment and uploads `all_scenarios.csv`, `all_configurations.csv`, and `winners.json`.

## Example run

Successful [Actions run #38849677456](https://github.com/evaristebernhard/fast_line/actions/runs/38849677456) produced:

| Arm | Tap offset | Added capacitor (kvar) | PV reactive Q (kvar) | Worst voltage margin (pu) | Min voltage (pu) | Max voltage (pu) |
|---|---:|---:|---:|---:|---:|---:|
| B0 fixed | 0 | 0 | 0 | -0.043228 | 0.906772 | 1.001946 |
| B1 tap + capacitor | 2 | 300 | 0 | -0.026475 | 0.923525 | 1.027029 |
| B2 tap + capacitor + PV Q | 2 | 300 | 150 | -0.023146 | 0.926854 | 1.032642 |

**Interpretation:** device coordination helps, but none of the configurations is fully voltage-compliant for the chosen three snapshots. This motivates separating achievable control improvements from unmet feeder reinforcement needs.

## Scope

These snapshots are **synthetic operating assumptions**, not rural utility measurements. This first experiment screens **voltage only**; line-specific current ratings, switching cost, battery dispatch constraints, inverter Q–V curve scheduling, and full uncertainty-envelope certification remain to be implemented. Do not present these numerical results as validation of the original continuous-capacity/Hessian algorithm, which is separate.
