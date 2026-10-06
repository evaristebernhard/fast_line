"""Direct OpenDSS IEEE-123 GFM/VSG coupling experiment.

Unlike the NumPy feeder prototypes, this file calls the actual OpenDSS
three-phase circuit at every dynamic evaluation.  The official IEEE123
OpenDSS circuit is compiled, four GFM voltage sources are attached at buses
13, 47, 76, and 108, and their positive-sequence source angles are updated
from the dynamic state.  OpenDSS then returns the GFM terminal powers and
currents and the complete feeder voltage profile.

The first direct-network stepper intentionally uses a black-box physical
balance predictor/correction rather than claiming that the reduced SAV
potential is already available for the full OpenDSS DAE:

    T_{n+1}-T_n
      + h*w_mid^T(P_e(delta_mid)+D*w_mid-p_control) = 0.

This isolates the real-network coupling and switching semantics.  Once this
benchmark is stable, the SAV auxiliary energy for the complete positive-
sequence DAE can be defined and compared on the same OpenDSS backend.

Requires ``opendssdirect.py`` and the data directory downloaded by the setup
step:

    data/ieee123_opendss/IEEE123Master.dss
    data/ieee123_opendss/IEEE123Loads.DSS
    data/ieee123_opendss/IEEELinecodes.DSS
    data/ieee123_opendss/IEEE123Regulators.DSS
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opendssdirect as dss
from scipy.integrate import solve_ivp


@dataclass
class State:
    delta: np.ndarray
    omega: np.ndarray


@dataclass
class NetworkPoint:
    p_electrical: np.ndarray
    current: np.ndarray
    min_voltage: float
    max_voltage: float


@dataclass
class StepDiagnostics:
    balance_residual: float
    predictor_defect: float
    correction_norm: float
    newton_iterations: int


class OpenDSS123GFM:
    """OpenDSS-backed GFM voltage-source interface."""

    def __init__(self, data_dir: Path,
                 gfm_buses=(13, 47, 76, 108),
                 initial_load_scale=0.60):
        self.data_dir = Path(data_dir).resolve()
        self.gfm_buses = tuple(int(x) for x in gfm_buses)
        self.n = len(self.gfm_buses)
        self.mass = np.array([0.050, 0.042, 0.038, 0.045])
        self.damping = np.array([0.16, 0.15, 0.14, 0.16])
        self.source_names = tuple(f"GFM{k}" for k in range(self.n))
        self.base_loads: dict[str, tuple[float, float]] = {}
        self.solve_count = 0
        self._compile(initial_load_scale)

    def _compile(self, initial_load_scale: float) -> None:
        master = self.data_dir / "IEEE123Master.dss"
        if not master.exists():
            raise FileNotFoundError(master)
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile [{master}]")
        dss.Text.Command("Set ControlMode=OFF")
        # Save nominal load values and create the islanded/restoration initial
        # condition by scaling all spot loads before adding GFM sources.
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            kw = float(dss.Loads.kW())
            kvar = float(dss.Loads.kvar())
            self.base_loads[name] = (kw, kvar)
            dss.Loads.kW(initial_load_scale * kw)
            dss.Loads.kvar(initial_load_scale * kvar)

        for k, bus in enumerate(self.gfm_buses):
            dss.Text.Command(
                f"New Vsource.{self.source_names[k]} phases=3 "
                f"Bus1={bus}.1.2.3 BasekV=4.16 pu=1 angle=0 "
                "R1=0.01 X1=0.05 R0=0.01 X0=0.05")
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError("initial OpenDSS IEEE-123 solve did not converge")

    def set_load_scale(self, scale: float) -> None:
        for name, (kw, kvar) in self.base_loads.items():
            dss.Loads.Name(name)
            dss.Loads.kW(scale * kw)
            dss.Loads.kvar(scale * kvar)

    def set_load_pickup(self, names: list[str], fraction: float) -> None:
        for name in names:
            if name not in self.base_loads:
                raise KeyError(f"unknown OpenDSS load {name}")
            kw, kvar = self.base_loads[name]
            dss.Loads.Name(name)
            dss.Loads.kW(fraction * kw)
            dss.Loads.kvar(fraction * kvar)

    def _set_angles(self, delta: np.ndarray) -> None:
        for name, angle in zip(self.source_names, delta):
            dss.Vsources.Name(name)
            dss.Vsources.AngleDeg(float(np.rad2deg(angle)))

    def evaluate(self, delta: np.ndarray) -> NetworkPoint:
        self._set_angles(delta)
        dss.Solution.Solve()
        self.solve_count += 1
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS IEEE-123 solve failed to converge")
        p_out = []
        current = []
        for name in self.source_names:
            dss.Circuit.SetActiveElement(f"Vsource.{name}")
            powers = np.asarray(dss.CktElement.Powers(), dtype=float)
            currents = np.asarray(dss.CktElement.CurrentsMagAng(), dtype=float)
            # CktElement.Powers is power entering the source element.  The
            # inverter electrical output is therefore the negative sum over
            # phase real powers.
            # OpenDSS reports kW; use a 100-MVA system base for the dynamic
            # equation so the inertia/damping parameters are in per-unit.
            p_out.append(-float(np.sum(powers[0::2])) / 1.0e5)
            current.append(float(np.max(currents[0::2])))
        magnitudes = np.asarray(dss.Circuit.AllBusMagPu(), dtype=float)
        magnitudes = magnitudes[magnitudes > 0.0]
        return NetworkPoint(np.asarray(p_out), np.asarray(current),
                            float(np.min(magnitudes)),
                            float(np.max(magnitudes)))


def physical_balance(new_state: State, old_state: State, h: float,
                     network: OpenDSS123GFM,
                     p_control: np.ndarray) -> float:
    midpoint = 0.5 * (old_state.delta + new_state.delta)
    midpoint_omega = 0.5 * (old_state.omega + new_state.omega)
    point = network.evaluate(midpoint)
    kinetic_jump = (
        0.5 * np.dot(network.mass * new_state.omega, new_state.omega)
        - 0.5 * np.dot(network.mass * old_state.omega, old_state.omega))
    return float(kinetic_jump + h * np.dot(
        midpoint_omega,
        point.p_electrical + network.damping * midpoint_omega - p_control))


def predictor(old: State, previous: State | None, h: float,
              network: OpenDSS123GFM,
              p_control: np.ndarray) -> tuple[State, float]:
    if previous is None:
        delta_star = old.delta + 0.5 * h * old.omega
    else:
        delta_star = 1.5 * old.delta - 0.5 * previous.delta
    point = network.evaluate(delta_star)
    a0 = 2.0 * network.mass / h + network.damping
    midpoint_omega = (
        2.0 * network.mass * old.omega / h
        + p_control - point.p_electrical) / a0
    new_state = State(
        old.delta + h * midpoint_omega,
        2.0 * midpoint_omega - old.omega)
    defect = physical_balance(new_state, old, h, network, p_control)
    return new_state, defect


def minimum_balance_correction(pred: State, old: State, h: float,
                               network: OpenDSS123GFM,
                               p_control: np.ndarray,
                               tol=2.0e-10) -> tuple[State, StepDiagnostics]:
    initial = physical_balance(pred, old, h, network, p_control)
    direction_delta = np.zeros(network.n)
    direction_omega = np.zeros(network.n)
    grad_delta = np.zeros(network.n)
    midpoint = 0.5 * (old.delta + pred.delta)
    midpoint_omega = 0.5 * (old.omega + pred.omega)
    midpoint_point = network.evaluate(midpoint)
    grad_omega = (network.mass * pred.omega
                  + 0.5 * h * (midpoint_point.p_electrical - p_control)
                  + h * network.damping * midpoint_omega)
    for i in range(network.n):
        eps = 2.0e-6 * max(1.0, abs(pred.delta[i]))
        plus = State(pred.delta.copy(), pred.omega.copy())
        minus = State(pred.delta.copy(), pred.omega.copy())
        plus.delta[i] += eps
        minus.delta[i] -= eps
        grad_delta[i] = (
            physical_balance(plus, old, h, network, p_control)
            - physical_balance(minus, old, h, network, p_control)) / (2.0 * eps)
    direction_delta = h * h * grad_delta / network.mass
    direction_omega = grad_omega / network.mass
    chi = float(np.dot(grad_delta, direction_delta)
                + np.dot(grad_omega, direction_omega))
    if abs(initial) <= tol:
        lam = 0.0
        iterations = 0
    else:
        if chi <= 1.0e-14:
            raise RuntimeError(f"OpenDSS balance direction degenerate: {chi:.3e}")

        def evaluate(lam_value):
            candidate = State(
                pred.delta + lam_value * direction_delta,
                pred.omega + lam_value * direction_omega)
            return physical_balance(candidate, old, h, network, p_control)

        lam = -initial / chi
        residual = initial
        for iterations in range(1, 16):
            residual = evaluate(lam)
            if abs(residual) <= tol:
                break
            eps = 2.0e-6 * max(1.0, abs(lam))
            deriv = (evaluate(lam + eps) - evaluate(lam - eps)) / (2.0 * eps)
            if abs(deriv) <= 1.0e-14:
                raise RuntimeError("OpenDSS scalar balance derivative vanished")
            candidate = lam - residual / deriv
            candidate_residual = evaluate(candidate)
            for _ in range(12):
                if abs(candidate_residual) <= abs(residual):
                    break
                candidate = 0.5 * (candidate + lam)
                candidate_residual = evaluate(candidate)
            lam = candidate
        else:
            raise RuntimeError(f"OpenDSS balance correction failed: {residual:.3e}")
    corrected = State(pred.delta + lam * direction_delta,
                      pred.omega + lam * direction_omega)
    residual = physical_balance(corrected, old, h, network, p_control)
    norm = float(np.sqrt(
        np.dot(network.mass * (corrected.delta - pred.delta),
               corrected.delta - pred.delta) / h**2
        + np.dot(network.mass * (corrected.omega - pred.omega),
                 corrected.omega - pred.omega)))
    return corrected, StepDiagnostics(abs(residual), 0.0, norm, iterations)


def run(method: str, data_dir: Path, h: float = 0.02,
        final_time: float = 1.5) -> dict[str, float]:
    network = OpenDSS123GFM(data_dir)
    initial = State(np.zeros(network.n), np.zeros(network.n))
    initial_point = network.evaluate(initial.delta)
    p_control = initial_point.p_electrical.copy()
    previous = None
    t = 0.0
    next_pickup = 1
    pickups = [
        (0.50, ["s47", "s48", "s49a", "s49b", "s49c"]),
        (1.00, ["s76a", "s76b", "s76c", "s77b"]),
    ]
    max_balance = 0.0
    max_defect = 0.0
    max_correction = 0.0
    min_frequency = np.inf
    min_voltage = initial_point.min_voltage
    max_current = float(np.max(initial_point.current))
    steps = 0
    start = time.perf_counter()

    while t < final_time - 1.0e-12:
        if next_pickup <= len(pickups) and t >= pickups[next_pickup - 1][0] - 1.0e-12:
            _, names = pickups[next_pickup - 1]
            network.set_load_pickup(names, 1.0)
            network.evaluate(initial.delta)
            previous = None
            next_pickup += 1
        hh = min(h, final_time - t)
        old = State(initial.delta.copy(), initial.omega.copy())
        pred, defect = predictor(old, previous, hh, network, p_control)
        if method == "direct-balance":
            state = pred
            diag = StepDiagnostics(0.0, abs(defect), 0.0, 0)
        else:
            state, diag = minimum_balance_correction(
                pred, old, hh, network, p_control)
            diag.predictor_defect = abs(defect)
        point = network.evaluate(state.delta)
        initial = state
        previous = old
        t += hh
        steps += 1
        min_frequency = min(min_frequency, float(np.min(state.omega)))
        min_voltage = min(min_voltage, point.min_voltage)
        max_current = max(max_current, float(np.max(point.current)))
        max_balance = max(max_balance, diag.balance_residual)
        max_defect = max(max_defect, diag.predictor_defect)
        max_correction = max(max_correction, diag.correction_norm)
    wall_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "method": method,
        "h": h,
        "steps": steps,
        "min_frequency": min_frequency,
        "min_voltage_pu": min_voltage,
        "max_gfm_current": max_current,
        "max_balance_residual": max_balance,
        "max_predictor_defect": max_defect,
        "max_correction_norm": max_correction,
        "open_dss_solves": network.solve_count,
        "wall_ms": wall_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/ieee123_opendss"))
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--output", type=Path,
                        default=Path("output/opendss_ieee123_gfm.csv"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    rows = [run(method, data_dir, args.h)
            for method in ("direct-balance", "minimum-balance")]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(*rows, sep="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
