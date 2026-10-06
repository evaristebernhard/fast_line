"""EC-SAV correction on the real three-phase OpenDSS IEEE-123 feeder.

The actual OpenDSS circuit remains the physical network.  A second independent
OpenDSS context is compiled from the same IEEE-123 files and made lossless by
zeroing the resistance matrices of non-switch lines.  For a vector of GFM
source angles delta, the two circuits provide

    P_e(delta)       actual R/X electrical power,
    P_B(delta)       lossless electrical power,
    P_G(delta)=P_e-P_B.

The lossless shadow power is integrated along the ray from zero to delta to
define a numerical conservative potential V_B.  The actual SAV split is then

    M omega_dot + D omega + r*b_B(delta) + P_G(delta) = p_control,
    b_B = grad(V_B)/sqrt(V_B+C),

with the lossless network supplying grad(V_B)=P_B.  The predictor is linear
in the SAV variables; the minimum-balance correction is one scalar Newton
solve using the actual OpenDSS physical balance.

This is the first version in the repository that uses the real three-phase
IEEE-123 network in the EC-SAV loop.  It is a positive-sequence GFM controller
interface: the GFM source angle is represented by the OpenDSS Vsource angle.
The current-limit experiment uses an accepted-step supervisory virtual-
impedance mode switch (increasing Vsource R1); it is intentionally explicit
about not being a detailed inner voltage/current loop.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import dss
import numpy as np


@dataclass
class State:
    delta: np.ndarray
    omega: np.ndarray
    r: float


@dataclass
class Point:
    p_actual: np.ndarray
    p_lossless: np.ndarray
    p_g: np.ndarray
    potential_b: float
    current: np.ndarray
    min_voltage: float
    max_voltage: float


@dataclass
class Diagnostics:
    balance_residual: float
    predictor_defect: float
    correction_norm: float
    newton_iterations: int


class OpenDSSNetwork:
    def __init__(self, data_dir: Path,
                 gfm_buses=(13, 47, 76, 108),
                 initial_load_scale=0.60,
                 current_limit_a=120.0):
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            # OpenDSS Compile/Redirect may change the process cwd.  Anchor
            # relative data paths to this source file so repeated candidate
            # simulations in one Python process remain reproducible.
            data_dir = Path(__file__).resolve().parent / data_dir
        self.data_dir = data_dir.resolve()
        self.gfm_buses = tuple(int(x) for x in gfm_buses)
        self.names = tuple(f"GFM{k}" for k in range(len(self.gfm_buses)))
        self.n = len(self.names)
        self.mass = np.array([0.050, 0.042, 0.038, 0.045])
        self.damping = np.array([0.16, 0.15, 0.14, 0.16])
        self.current_limit_a = (None if current_limit_a is None
                                else float(current_limit_a))
        self.base_r1 = 0.01
        self.max_r1 = 2.0
        self.source_r1 = np.full(self.n, self.base_r1)
        self.last_delta = np.zeros(self.n)
        self.initial_load_scale = float(initial_load_scale)
        self.actual = self._new_context(lossless=False, load_scale=initial_load_scale)
        self.shadow = self._new_context(lossless=True, load_scale=initial_load_scale)
        scaled_loads = self._read_base_loads(self.actual)
        self.base_loads = {
            name: (kw / self.initial_load_scale,
                   kvar / self.initial_load_scale)
            for name, (kw, kvar) in scaled_loads.items()
        }
        self.solve_count = 0
        self.shadow_solves = 0
        self.actual_solves = 0
        self.quad_nodes, self.quad_weights = np.polynomial.legendre.leggauss(4)
        self.quad_nodes = 0.5 * (self.quad_nodes + 1.0)
        self.quad_weights = 0.5 * self.quad_weights

    def _new_context(self, *, lossless: bool, load_scale: float):
        ctx = dss.DSS.NewContext().to_opendssdirect()
        ctx.Basic.ClearAll()
        ctx.Text.Command(f"Compile [{self.data_dir / 'IEEE123Master.dss'}]")
        ctx.Text.Command("Set ControlMode=OFF")
        for name in ctx.Loads.AllNames():
            ctx.Loads.Name(name)
            ctx.Loads.kW(load_scale * float(ctx.Loads.kW()))
            ctx.Loads.kvar(load_scale * float(ctx.Loads.kvar()))
        if lossless:
            for name in ctx.Lines.AllNames():
                ctx.Lines.Name(name)
                if name.lower().startswith("sw"):
                    # The official switch elements have X=0.  Keep their
                    # small resistance and add a tiny X so Yprim remains
                    # invertible in the lossless shadow context.
                    nphase = int(ctx.Lines.Phases())
                    ctx.Lines.XMatrix([1.0e-5] * (nphase * nphase))
                else:
                    nphase = int(ctx.Lines.Phases())
                    ctx.Lines.RMatrix([0.0] * (nphase * nphase))
        for k, bus in enumerate(self.gfm_buses):
            r1 = 0.0 if lossless else self.base_r1
            ctx.Text.Command(
                f"New Vsource.{self.names[k]} phases=3 Bus1={bus}.1.2.3 "
                f"BasekV=4.16 pu=1 angle=0 R1={r1} X1=0.05 "
                f"R0={r1} X0=0.05")
        ctx.Solution.Solve()
        if not ctx.Solution.Converged():
            raise RuntimeError(f"OpenDSS {'lossless' if lossless else 'actual'} "
                               "initial solve did not converge")
        return ctx

    @staticmethod
    def _read_base_loads(ctx):
        result = {}
        for name in ctx.Loads.AllNames():
            ctx.Loads.Name(name)
            result[name] = (float(ctx.Loads.kW()), float(ctx.Loads.kvar()))
        return result

    def _set_actual_r1(self, index, value):
        value = float(value)
        self.actual.Text.Command(
            f"Edit Vsource.{self.names[index]} R1={value} R0={value}")
        self.source_r1[index] = value

    def update_current_limit(self, currents):
        """Latch virtual-impedance limiting when a GFM exceeds its limit.

        The update is performed only after an accepted time step.  This makes
        the R1 change a controller-mode switch, so the caller can reset the
        SAV auxiliary variable and extrapolation history at that point.
        """
        if self.current_limit_a is None:
            return False
        changed = False
        currents = np.asarray(currents, dtype=float)
        for _ in range(12):
            overloaded = np.flatnonzero(currents > self.current_limit_a)
            if overloaded.size == 0:
                break
            for index in overloaded:
                if self.source_r1[index] >= self.max_r1 - 1.0e-12:
                    continue
                # Increase virtual resistance geometrically, then re-solve the
                # same actual feeder state.  This is a supervisory current
                # limiter, not a voltage clipping rule.
                new_r1 = min(self.max_r1,
                             max(self.source_r1[index] + 0.05,
                                 1.6 * self.source_r1[index]))
                self._set_actual_r1(index, new_r1)
                self._set_angles(self.actual, self.last_delta)
                self.actual.Solution.Solve()
                self.actual_solves += 1
                if not self.actual.Solution.Converged():
                    raise RuntimeError("OpenDSS solve failed in GFM current limiter")
                _, currents, _, _ = self._powers(self.actual)
                changed = True
        return changed

    @property
    def limited_gfms(self):
        return int(np.count_nonzero(self.source_r1 > self.base_r1 + 1.0e-12))

    def set_load_pickup(self, names):
        for ctx in (self.actual, self.shadow):
            for name in names:
                ctx.Loads.Name(name)
                kw, kvar = self.base_loads[name]
                ctx.Loads.kW(kw)
                ctx.Loads.kvar(kvar)

    def open_switch(self, name):
        """Open an official IEEE-123 switch in both full network contexts."""
        for ctx in (self.actual, self.shadow):
            ctx.Text.Command(f"Open Line.{name} 1")
            ctx.Solution.Solve()
            if not ctx.Solution.Converged():
                raise RuntimeError(f"OpenDSS solve failed after opening {name}")

    def _set_angles(self, ctx, delta):
        for name, angle in zip(self.names, delta):
            ctx.Vsources.Name(name)
            ctx.Vsources.AngleDeg(float(np.rad2deg(angle)))

    def _powers(self, ctx):
        values = []
        currents = []
        for name in self.names:
            ctx.Circuit.SetActiveElement(f"Vsource.{name}")
            powers = np.asarray(ctx.CktElement.Powers(), dtype=float)
            current = np.asarray(ctx.CktElement.CurrentsMagAng(), dtype=float)
            values.append(-float(np.sum(powers[0::2])) / 1.0e5)
            currents.append(float(np.max(current[0::2])))
        mags = np.asarray(ctx.Circuit.AllBusMagPu(), dtype=float)
        mags = mags[mags > 0.0]
        return np.asarray(values), np.asarray(currents), float(np.min(mags)), float(np.max(mags))

    def _shadow_power(self, delta):
        self._set_angles(self.shadow, delta)
        self.shadow.Solution.Solve()
        self.shadow_solves += 1
        if not self.shadow.Solution.Converged():
            raise RuntimeError("lossless OpenDSS shadow solve failed")
        p, _, _, _ = self._powers(self.shadow)
        return p

    def _actual_power(self, delta):
        self.last_delta = np.asarray(delta, dtype=float).copy()
        self._set_angles(self.actual, delta)
        self.actual.Solution.Solve()
        self.actual_solves += 1
        if not self.actual.Solution.Converged():
            raise RuntimeError("actual OpenDSS solve failed")
        p, current, vmin, vmax = self._powers(self.actual)
        return p, current, vmin, vmax

    def potential(self, delta):
        total = 0.0
        for node, weight in zip(self.quad_nodes, self.quad_weights):
            total += weight * np.dot(self._shadow_power(node * delta), delta)
        return float(total)

    def evaluate(self, delta):
        p_actual, current, vmin, vmax = self._actual_power(delta)
        p_lossless = self._shadow_power(delta)
        potential = self.potential(delta)
        self.solve_count += 1
        return Point(p_actual, p_lossless, p_actual - p_lossless,
                     potential, current, vmin, vmax)


def balance_residual(new_state, old_state, h, network, p_control,
                     old_point=None):
    if old_point is None:
        old_point = network.evaluate(old_state.delta)
    new_point = network.evaluate(new_state.delta)
    midpoint = 0.5 * (old_state.delta + new_state.delta)
    midpoint_omega = 0.5 * (old_state.omega + new_state.omega)
    mid_point = network.evaluate(midpoint)
    old_h = (0.5 * np.dot(network.mass * old_state.omega, old_state.omega)
             + old_point.potential_b - np.dot(p_control, old_state.delta))
    new_h = (0.5 * np.dot(network.mass * new_state.omega, new_state.omega)
             + new_point.potential_b - np.dot(p_control, new_state.delta))
    return float(new_h - old_h + h * np.dot(
        midpoint_omega,
        network.damping * midpoint_omega + mid_point.p_g))


def predictor(old, previous, h, network, p_control, C):
    if previous is None:
        delta_star = old.delta + 0.5 * h * old.omega
    else:
        delta_star = 1.5 * old.delta - 0.5 * previous.delta
    point = network.evaluate(delta_star)
    qsq = point.potential_b + C
    if qsq <= 0.0:
        raise RuntimeError(f"lossless shadow potential+C is non-positive: {qsq}")
    b = point.p_lossless / np.sqrt(qsq)
    a0 = 2.0 * network.mass / h + network.damping
    v = (2.0 * network.mass * old.omega / h
         + p_control - point.p_g - old.r * b) / a0
    new_state = State(old.delta + h * v, 2.0 * v - old.omega,
                      old.r + 0.5 * h * np.dot(b, v))
    defect_point = network.evaluate(new_state.delta)
    defect = defect_point.potential_b + C - new_state.r**2
    return new_state, float(defect)


def correction(pred, old, h, network, p_control, C, old_point,
               tol=2.0e-9):
    initial = balance_residual(pred, old, h, network, p_control, old_point)
    midpoint = 0.5 * (old.delta + pred.delta)
    midpoint_point = network.evaluate(midpoint)
    midpoint_omega = 0.5 * (old.omega + pred.omega)
    grad_omega = (network.mass * pred.omega
                  + h * network.damping * midpoint_omega
                  + 0.5 * h * midpoint_point.p_g)
    grad_delta = np.zeros(network.n)
    for i in range(network.n):
        eps = 2.0e-6 * max(1.0, abs(pred.delta[i]))
        plus = State(pred.delta.copy(), pred.omega.copy(), pred.r)
        minus = State(pred.delta.copy(), pred.omega.copy(), pred.r)
        plus.delta[i] += eps
        minus.delta[i] -= eps
        grad_delta[i] = (
            balance_residual(plus, old, h, network, p_control, old_point)
            - balance_residual(minus, old, h, network, p_control, old_point)
        ) / (2.0 * eps)
    p_delta = h * h * grad_delta / network.mass
    p_omega = grad_omega / network.mass
    chi = float(np.dot(grad_delta, p_delta) + np.dot(grad_omega, p_omega))
    if abs(initial) <= tol:
        lam = 0.0
        iterations = 0
    else:
        if chi <= 1.0e-14:
            raise RuntimeError(f"EC-SAV OpenDSS correction degenerate: {chi:.3e}")

        def evaluate(lam_value):
            candidate = State(pred.delta + lam_value * p_delta,
                              pred.omega + lam_value * p_omega, pred.r)
            return balance_residual(candidate, old, h, network, p_control, old_point)

        lam = -initial / chi
        residual = initial
        for iterations in range(1, 16):
            residual = evaluate(lam)
            if abs(residual) <= tol:
                break
            eps = 2.0e-6 * max(1.0, abs(lam))
            derivative = (evaluate(lam + eps) - evaluate(lam - eps)) / (2.0 * eps)
            if abs(derivative) <= 1.0e-14:
                raise RuntimeError("EC-SAV OpenDSS scalar derivative vanished")
            candidate = lam - residual / derivative
            candidate_residual = evaluate(candidate)
            for _ in range(12):
                if abs(candidate_residual) <= abs(residual):
                    break
                candidate = 0.5 * (candidate + lam)
                candidate_residual = evaluate(candidate)
            lam = candidate
        else:
            raise RuntimeError(f"EC-SAV OpenDSS correction failed: {residual:.3e}")
    delta = pred.delta + lam * p_delta
    omega = pred.omega + lam * p_omega
    point = network.evaluate(delta)
    corrected = State(delta, omega, float(np.sqrt(point.potential_b + C)))
    residual = balance_residual(corrected, old, h, network, p_control, old_point)
    norm = float(np.sqrt(
        np.dot(network.mass * (delta - pred.delta), delta - pred.delta) / h**2
        + np.dot(network.mass * (omega - pred.omega), omega - pred.omega)))
    return corrected, Diagnostics(abs(residual), 0.0, norm, iterations)


def run(method, data_dir, h=0.1, final_time=1.5, trajectory=None,
        current_limit_a=120.0):
    network = OpenDSSNetwork(data_dir, current_limit_a=current_limit_a)
    C = 30.0
    state = State(np.zeros(network.n), np.zeros(network.n), np.sqrt(C))
    initial = network.evaluate(state.delta)
    p_control = initial.p_actual.copy()
    previous = None
    events = [
        (0.25, "switch", "sw2"),
        (0.50, "pickup", ["s47", "s48", "s49a", "s49b", "s49c"]),
        (1.00, "pickup", ["s76a", "s76b", "s76c", "s77b"]),
    ]
    next_event = 0
    t = 0.0
    steps = 0
    max_balance = 0.0
    max_defect = 0.0
    max_correction = 0.0
    min_frequency_deviation_hz = 0.0
    max_abs_frequency_deviation_hz = 0.0
    max_angle_spread_deg = 0.0
    min_voltage = initial.min_voltage
    max_current = float(np.max(initial.current))
    start = time.perf_counter()
    while t < final_time - 1.0e-12:
        event_label = ""
        if next_event < len(events) and t >= events[next_event][0] - 1.0e-12:
            event_time, kind, payload = events[next_event]
            if kind == "switch":
                network.open_switch(payload)
                event_label = f"open:{payload}@{event_time:g}s"
            else:
                network.set_load_pickup(payload)
                event_label = ("pickup:" + "+".join(payload)
                               + f"@{event_time:g}s")
            event_point = network.evaluate(state.delta)
            state.r = float(np.sqrt(event_point.potential_b + C))
            previous = None
            next_event += 1
        hh = min(h, final_time - t)
        if next_event < len(events):
            hh = min(hh, events[next_event][0] - t)
        if hh <= 1.0e-12:
            continue
        old = State(state.delta.copy(), state.omega.copy(), state.r)
        old_point = network.evaluate(old.delta)
        pred, defect = predictor(old, previous, hh, network, p_control, C)
        if method == "sav":
            state = pred
            diag = Diagnostics(
                abs(balance_residual(state, old, hh, network,
                                     p_control, old_point)),
                abs(defect), 0.0, 0)
        else:
            state, diag = correction(pred, old, hh, network, p_control, C, old_point)
            diag.predictor_defect = abs(defect)
        previous = old
        t += hh
        steps += 1
        point = network.evaluate(state.delta)
        if network.update_current_limit(point.current):
            # The virtual-impedance change is a controller mode switch.  Keep
            # the physical state continuous, but restart the SAV bookkeeping
            # against the new actual-network balance.
            point = network.evaluate(state.delta)
            state.r = float(np.sqrt(point.potential_b + C))
            previous = None
            event_label = (event_label + ";" if event_label
                            else "") + "current-limit"
        frequency_deviation_hz = state.omega / (2.0 * np.pi)
        min_frequency_deviation_hz = min(
            min_frequency_deviation_hz, float(np.min(frequency_deviation_hz)))
        max_abs_frequency_deviation_hz = max(
            max_abs_frequency_deviation_hz,
            float(np.max(np.abs(frequency_deviation_hz))))
        max_angle_spread_deg = max(
            max_angle_spread_deg,
            float(np.rad2deg(np.max(state.delta) - np.min(state.delta))))
        min_voltage = min(min_voltage, point.min_voltage)
        max_current = max(max_current, float(np.max(point.current)))
        max_balance = max(max_balance, diag.balance_residual)
        max_defect = max(max_defect, diag.predictor_defect)
        max_correction = max(max_correction, diag.correction_norm)
        if trajectory is not None:
            trajectory.append({
                "method": method,
                "time_s": t,
                "min_frequency_deviation_hz": float(np.min(frequency_deviation_hz)),
                "max_abs_frequency_deviation_hz": float(np.max(np.abs(frequency_deviation_hz))),
                "angle_spread_deg": float(np.rad2deg(
                    np.max(state.delta) - np.min(state.delta))),
                "min_voltage_pu": point.min_voltage,
                "max_gfm_current_A": float(np.max(point.current)),
                "balance_residual": diag.balance_residual,
                "predictor_defect": diag.predictor_defect,
                "correction_norm": diag.correction_norm,
                "limited_gfm_count": network.limited_gfms,
                "max_virtual_r1": float(np.max(network.source_r1)),
                "event": event_label,
            })
    return {
        "method": method,
        "h": h,
        "current_limit_a": ("none" if network.current_limit_a is None
                             else network.current_limit_a),
        "steps": steps,
        "min_voltage_pu": min_voltage,
        "max_gfm_current_A": max_current,
        "max_balance_residual": max_balance,
        "max_predictor_defect": max_defect,
        "max_correction_norm": max_correction,
        "min_frequency_deviation_hz": min_frequency_deviation_hz,
        "max_abs_frequency_deviation_hz": max_abs_frequency_deviation_hz,
        "max_angle_spread_deg": max_angle_spread_deg,
        "limited_gfm_count": network.limited_gfms,
        "max_virtual_r1": float(np.max(network.source_r1)),
        "actual_solves": network.actual_solves,
        "shadow_solves": network.shadow_solves,
        "wall_ms": 1000.0 * (time.perf_counter() - start),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ieee123_opendss"))
    parser.add_argument("--h", type=float, default=0.1)
    parser.add_argument("--current-limit-a", type=float, default=120.0,
                        help="GFM current limit in A; use 0 to disable")
    parser.add_argument("--output", type=Path,
                        default=Path("output/opendss_ieee123_ecsav.csv"))
    parser.add_argument("--trajectory-output", type=Path,
                        default=Path("output/opendss_ieee123_ecsav_trajectory.csv"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    trajectory_output = args.trajectory_output.resolve()
    rows = []
    trajectories = []
    for method in ("sav", "minimum-balance"):
        trace = []
        limit = None if args.current_limit_a <= 0.0 else args.current_limit_a
        rows.append(run(method, data_dir, args.h, trajectory=trace,
                        current_limit_a=limit))
        trajectories.extend(trace)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectories[0].keys())
        writer.writeheader()
        writer.writerows(trajectories)
    print(*rows, sep="\n")
    print(f"wrote {output}")
    print(f"wrote {trajectory_output}")


if __name__ == "__main__":
    main()
