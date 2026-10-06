"""Hybrid EC-SAV experiments suggested by 1.md and 2.md.

The script tests two claims that were not exercised by the earlier EC-SAV
prototype:

1. The SAV consistency defect is O(h^3) at a regular point and O(h^4) at a
   non-degenerate turning point.
2. A hybrid kinetic/potential correction removes that scalar defect while
   preserving the SAV predictor's original-energy dissipation target.

The second experiment applies the method to the damped IEEE 9-bus classical
three-machine model with a bus-7 balanced fault and line 7--8 clearing.
Only NumPy and SciPy are required.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import ieee9_ecsav_cct as case9


def total_potential(delta, net, pm):
    """Full mechanical potential: network potential minus Pm^T delta."""
    return net.potential(delta) - np.dot(pm, delta)


def total_gradient(delta, net, pm):
    return net.gradient(delta) - pm


def physical_energy(delta, omega, net, mass, pm):
    return 0.5 * np.dot(mass * omega, omega) + total_potential(delta, net, pm)


@dataclass
class Predictor:
    delta: np.ndarray
    omega: np.ndarray
    r: float
    midpoint_velocity: np.ndarray
    delta_star: np.ndarray
    epsilon: float
    modified_balance_defect: float


@dataclass
class Correction:
    delta: np.ndarray
    omega: np.ndarray
    r: float
    branch: str
    epsilon: float
    kappa: float
    chi_delta: float
    scale: float
    lam: float
    balance_defect: float


def sav_predictor(delta_prev, delta, omega, r, h, net, mass, damping, pm, C):
    """Second-order linearly implicit SAV predictor from 1.md/2.md."""
    if delta_prev is None:
        delta_star = delta + 0.5 * h * omega
    else:
        delta_star = 1.5 * delta - 0.5 * delta_prev

    qstar_sq = total_potential(delta_star, net, pm) + C
    if qstar_sq <= 0.0:
        raise RuntimeError(
            f"SAV square root lost positivity: U+C={qstar_sq:.6e}")
    b = total_gradient(delta_star, net, pm) / np.sqrt(qstar_sq)

    a0 = 2.0 * mass / h + damping
    rhs = 2.0 * mass * omega / h - r * b
    inv = 1.0 / a0
    y = inv * rhs
    inv_b = inv * b
    alpha = h / 4.0
    v = y - inv_b * (alpha * np.dot(b, y)) / (
        1.0 + alpha * np.dot(b, inv_b))

    delta_tilde = delta + h * v
    omega_tilde = 2.0 * v - omega
    r_tilde = r + 0.5 * h * np.dot(b, v)
    epsilon = total_potential(delta_tilde, net, pm) + C - r_tilde**2

    modified_old = 0.5 * np.dot(mass * omega, omega) + r**2 - C
    modified_new = (
        0.5 * np.dot(mass * omega_tilde, omega_tilde) + r_tilde**2 - C)
    modified_balance_defect = (
        modified_new - modified_old + h * np.dot(damping * v, v))
    return Predictor(
        delta_tilde, omega_tilde, float(r_tilde), v, delta_star,
        float(epsilon), float(modified_balance_defect))


def correction_indicators(pred, mass, net, pm):
    # COI projection is appropriate only for a rotationally invariant
    # multimachine model.  An SMIB angle is measured against an infinite bus,
    # so its single velocity component must not be projected out.
    use_coi = len(mass) > 1 and abs(np.sum(pm)) <= 1.0e-12
    center = np.dot(mass, pred.omega) / np.sum(mass) if use_coi else 0.0
    relative = pred.omega - center
    kappa = float(np.dot(mass * relative, relative))
    grad = total_gradient(pred.delta, net, pm)
    chi_delta = float(np.dot(grad / mass, grad))
    return center, relative, grad, kappa, chi_delta


def kinetic_correction(pred, Hprev, h, net, mass, damping, pm, C):
    center, relative, grad, kappa, chi_delta = correction_indicators(
        pred, mass, net, pm)
    del grad
    tol = 2.0e-14 * max(1.0, abs(Hprev))
    if kappa <= tol:
        if abs(pred.epsilon) <= tol:
            scale = 1.0
        else:
            raise RuntimeError(
                f"kinetic correction is singular: kappa={kappa:.3e}, "
                f"epsilon={pred.epsilon:.3e}")
    else:
        radicand = 1.0 - 2.0 * pred.epsilon / kappa
        if radicand < -2.0e-13:
            raise RuntimeError(
                f"kinetic correction has negative radicand {radicand:.3e}")
        scale = np.sqrt(max(0.0, radicand))

    omega = center + scale * relative
    delta = pred.delta.copy()
    r = np.sqrt(total_potential(delta, net, pm) + C)
    target = Hprev - h * np.dot(damping * pred.midpoint_velocity,
                                pred.midpoint_velocity)
    defect = physical_energy(delta, omega, net, mass, pm) - target
    return Correction(delta, omega, float(r), "kinetic", pred.epsilon,
                      kappa, chi_delta, float(scale), 0.0, float(defect))


def potential_correction(pred, Hprev, h, net, mass, damping, pm, C):
    center, relative, grad, kappa, chi_delta = correction_indicators(
        pred, mass, net, pm)
    del center, relative
    tol = 2.0e-14 * max(1.0, abs(Hprev))
    if chi_delta <= tol:
        if abs(pred.epsilon) <= tol:
            lam = 0.0
        else:
            raise RuntimeError(
                f"potential correction is singular: chi={chi_delta:.3e}, "
                f"epsilon={pred.epsilon:.3e}")
    else:
        direction = grad / mass
        lam = -pred.epsilon / chi_delta
        converged = False
        for _ in range(20):
            trial = pred.delta + lam * direction
            phi = total_potential(trial, net, pm) + C - pred.r**2
            if abs(phi) <= 5.0e-14 * max(1.0, pred.r**2):
                converged = True
                break
            deriv = np.dot(total_gradient(trial, net, pm), direction)
            if abs(deriv) <= 1.0e-16:
                break
            step = phi / deriv
            # The desired root is the local root near zero.  Limiting a wild
            # Newton update keeps the iteration on that branch.
            limit = 4.0 * max(abs(lam), abs(pred.epsilon) / chi_delta, 1.0e-12)
            lam -= np.clip(step, -limit, limit)
        if not converged:
            trial = pred.delta + lam * direction
            phi = total_potential(trial, net, pm) + C - pred.r**2
            if abs(phi) > 2.0e-11 * max(1.0, pred.r**2):
                raise RuntimeError(
                    f"potential correction Newton failed: phi={phi:.3e}")

    direction = grad / mass if chi_delta > tol else np.zeros_like(grad)
    delta = pred.delta + lam * direction
    omega = pred.omega.copy()
    r = np.sqrt(total_potential(delta, net, pm) + C)
    target = Hprev - h * np.dot(damping * pred.midpoint_velocity,
                                pred.midpoint_velocity)
    defect = physical_energy(delta, omega, net, mass, pm) - target
    return Correction(delta, omega, float(r), "potential", pred.epsilon,
                      kappa, chi_delta, 1.0, float(lam), float(defect))


def hybrid_correction(pred, Hprev, h, net, mass, damping, pm, C,
                      turning_factor=4.0):
    """Choose a well-conditioned branch and fall back to the other branch."""
    _, _, _, kappa, chi_delta = correction_indicators(pred, mass, net, pm)
    # h^2*chi_delta has the scale of the relative kinetic indicator produced
    # over one step.  This sends non-degenerate turning points to the
    # potential branch while retaining kinetic correction at regular points.
    prefer_kinetic = kappa > turning_factor * h * h * chi_delta
    order = (kinetic_correction, potential_correction) if prefer_kinetic else (
        potential_correction, kinetic_correction)
    errors = []
    for correct in order:
        try:
            return correct(pred, Hprev, h, net, mass, damping, pm, C)
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("both Hybrid EC-SAV branches failed: " + " | ".join(errors))


class SMIBPotential:
    def __init__(self, coupling, mechanical_power):
        self.coupling = float(coupling)
        self.mechanical_power = float(mechanical_power)

    def potential(self, delta):
        return float(-self.coupling * np.cos(delta[0]))

    def gradient(self, delta):
        return np.array([self.coupling * np.sin(delta[0])])


def observed_orders(values, steps):
    values = np.abs(np.asarray(values, dtype=float))
    steps = np.asarray(steps, dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(1, len(values)):
        if values[i] > 0.0 and values[i - 1] > 0.0:
            out[i] = np.log(values[i - 1] / values[i]) / np.log(
                steps[i - 1] / steps[i])
    return out


def smib_local_order_experiment():
    mass = np.array([1.3])
    damping = np.array([0.2])
    pm = np.array([0.4])
    net = SMIBPotential(coupling=1.1, mechanical_power=pm[0])
    C = 3.0
    steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
    cases = {
        "regular": (np.array([0.8]), np.array([0.4])),
        "turning": (np.array([0.8]), np.array([0.0])),
    }
    rows = []
    for case_name, (delta, omega) in cases.items():
        eps_values = []
        kinetic_changes = []
        potential_changes = []
        auto_branches = []
        energy_defects = []
        identity_errors = []
        for h in steps:
            def rhs(_t, y):
                return np.array([
                    y[1],
                    (-damping[0] * y[1]
                     - total_gradient(np.array([y[0]]), net, pm)[0]) / mass[0],
                ])

            back = solve_ivp(rhs, (0.0, -h), np.r_[delta, omega],
                             method="DOP853", rtol=2e-13, atol=2e-15,
                             max_step=h / 20.0)
            delta_prev = back.y[:1, -1]
            r = np.sqrt(total_potential(delta, net, pm) + C)
            Hprev = physical_energy(delta, omega, net, mass, pm)
            pred = sav_predictor(delta_prev, delta, omega, r, h, net,
                                 mass, damping, pm, C)
            kinetic = kinetic_correction(pred, Hprev, h, net, mass,
                                         damping, pm, C)
            potential = potential_correction(pred, Hprev, h, net, mass,
                                             damping, pm, C)
            auto = hybrid_correction(pred, Hprev, h, net, mass,
                                     damping, pm, C)
            predictor_physical_residual = (
                physical_energy(pred.delta, pred.omega, net, mass, pm)
                - Hprev
                + h * np.dot(damping * pred.midpoint_velocity,
                             pred.midpoint_velocity))

            eps_values.append(abs(pred.epsilon))
            kinetic_changes.append(
                np.sqrt(np.dot(mass * (kinetic.omega - pred.omega),
                               kinetic.omega - pred.omega)))
            potential_changes.append(
                np.sqrt(np.dot(mass * (potential.delta - pred.delta),
                               potential.delta - pred.delta)))
            auto_branches.append(auto.branch)
            energy_defects.append(abs(auto.balance_defect))
            identity_errors.append(predictor_physical_residual - pred.epsilon)

        eps_orders = observed_orders(eps_values, steps)
        kinetic_orders = observed_orders(kinetic_changes, steps)
        potential_orders = observed_orders(potential_changes, steps)
        for i, h in enumerate(steps):
            rows.append({
                "case": case_name,
                "h": h,
                "epsilon": eps_values[i],
                "epsilon_order": eps_orders[i],
                "kinetic_state_correction": kinetic_changes[i],
                "kinetic_correction_order": kinetic_orders[i],
                "potential_state_correction": potential_changes[i],
                "potential_correction_order": potential_orders[i],
                "auto_branch": auto_branches[i],
                "auto_energy_balance_defect": energy_defects[i],
                "defect_identity_error": identity_errors[i],
            })
    return rows


def smib_global_convergence_experiment():
    mass = np.array([1.3])
    damping = np.array([0.2])
    pm = np.array([0.4])
    net = SMIBPotential(coupling=1.1, mechanical_power=pm[0])
    C = 3.0
    delta0 = np.array([0.8])
    omega0 = np.array([0.4])
    final_time = 1.0

    def rhs(_t, y):
        return np.array([
            y[1],
            (-damping[0] * y[1]
             - total_gradient(np.array([y[0]]), net, pm)[0]) / mass[0],
        ])

    reference = solve_ivp(
        rhs, (0.0, final_time), np.r_[delta0, omega0], method="DOP853",
        rtol=2.0e-13, atol=2.0e-15, max_step=2.0e-4).y[:, -1]
    rows = []
    steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
    for method in ("sav", "hybrid"):
        errors = []
        branch_counts = []
        balance_defects = []
        for h in steps:
            delta_prev = None
            delta = delta0.copy()
            omega = omega0.copy()
            r = np.sqrt(total_potential(delta, net, pm) + C)
            stats = init_stats()
            max_balance = 0.0
            for _ in range(int(round(final_time / h))):
                Hprev = physical_energy(delta, omega, net, mass, pm)
                pred = sav_predictor(delta_prev, delta, omega, r, h, net,
                                     mass, damping, pm, C)
                old_delta = delta
                if method == "sav":
                    delta, omega, r = pred.delta, pred.omega, pred.r
                else:
                    corrected = hybrid_correction(
                        pred, Hprev, h, net, mass, damping, pm, C)
                    update_stats(stats, pred, corrected)
                    delta, omega, r = (
                        corrected.delta, corrected.omega, corrected.r)
                    max_balance = max(max_balance, abs(corrected.balance_defect))
                delta_prev = old_delta
            error = np.linalg.norm(np.r_[delta, omega] - reference)
            errors.append(error)
            branch_counts.append((stats["kinetic_steps"],
                                  stats["potential_steps"]))
            balance_defects.append(max_balance)
        orders = observed_orders(errors, steps)
        for i, h in enumerate(steps):
            rows.append({
                "method": method,
                "h": h,
                "state_error": errors[i],
                "observed_order": orders[i],
                "kinetic_steps": branch_counts[i][0],
                "potential_steps": branch_counts[i][1],
                "max_energy_balance_defect": balance_defects[i],
            })
    return rows


def init_stats():
    return {
        "steps": 0,
        "kinetic_steps": 0,
        "potential_steps": 0,
        "max_abs_epsilon": 0.0,
        "max_abs_balance_defect": 0.0,
        "max_abs_modified_defect": 0.0,
        "min_kinetic_scale": np.inf,
        "max_kinetic_scale": -np.inf,
        "max_abs_lambda": 0.0,
        "switch_jump_defect": 0.0,
    }


def update_stats(stats, pred, corrected):
    stats["steps"] += 1
    stats[f"{corrected.branch}_steps"] += 1
    stats["max_abs_epsilon"] = max(stats["max_abs_epsilon"], abs(pred.epsilon))
    stats["max_abs_balance_defect"] = max(
        stats["max_abs_balance_defect"], abs(corrected.balance_defect))
    stats["max_abs_modified_defect"] = max(
        stats["max_abs_modified_defect"], abs(pred.modified_balance_defect))
    if corrected.branch == "kinetic":
        stats["min_kinetic_scale"] = min(
            stats["min_kinetic_scale"], corrected.scale)
        stats["max_kinetic_scale"] = max(
            stats["max_kinetic_scale"], corrected.scale)
    else:
        stats["max_abs_lambda"] = max(
            stats["max_abs_lambda"], abs(corrected.lam))


def advance_hybrid(delta_prev, delta, omega, r, duration, h, net, mass,
                   damping, pm, C, stats, track_spread=False,
                   stop_spread=np.inf):
    elapsed = 0.0
    max_spread = float(np.max(delta) - np.min(delta))
    while elapsed < duration - 1.0e-14:
        hh = min(h, duration - elapsed)
        Hprev = physical_energy(delta, omega, net, mass, pm)
        pred = sav_predictor(delta_prev, delta, omega, r, hh, net,
                             mass, damping, pm, C)
        corrected = hybrid_correction(pred, Hprev, hh, net, mass,
                                      damping, pm, C)
        update_stats(stats, pred, corrected)
        delta_prev, delta = delta, corrected.delta
        omega, r = corrected.omega, corrected.r
        if track_spread:
            max_spread = max(
                max_spread, float(np.max(delta) - np.min(delta)))
            if max_spread >= stop_spread:
                break
        elapsed += hh
    return delta_prev, delta, omega, r, max_spread


def simulate_ieee9_hybrid(tc, h, nets, delta0, pm, mass, damping,
                          C=10.0, post_time=2.0):
    delta = delta0.copy()
    delta_prev = delta0.copy()
    omega = np.zeros_like(delta)
    fault = nets["fault"]
    r = np.sqrt(total_potential(delta, fault, pm) + C)
    stats = init_stats()
    delta_prev, delta, omega, r, fault_spread = advance_hybrid(
        delta_prev, delta, omega, r, tc, h, fault, mass, damping, pm, C,
        stats, track_spread=True, stop_spread=np.pi)
    if fault_spread >= np.pi:
        return False, fault_spread, stats

    # Exact state continuity plus the physical potential jump at clearing.
    Hminus = physical_energy(delta, omega, fault, mass, pm)
    jump_exact = (total_potential(delta, nets["post"], pm)
                  - total_potential(delta, fault, pm))
    Hplus = physical_energy(delta, omega, nets["post"], mass, pm)
    stats["switch_jump_defect"] = abs((Hplus - Hminus) - jump_exact)
    post = nets["post"]
    r = np.sqrt(total_potential(delta, post, pm) + C)
    delta_prev, delta, omega, r, post_spread = advance_hybrid(
        delta_prev, delta, omega, r, post_time, h, post, mass, damping,
        pm, C, stats, track_spread=True, stop_spread=np.pi)
    max_spread = max(fault_spread, post_spread)
    stable = np.isfinite(max_spread) and max_spread < np.pi
    return stable, max_spread, stats


def hybrid_cct(h, nets, delta0, pm, mass, damping, cct_ref):
    lo, hi = 0.0, max(0.5, 2.0 * cct_ref)
    while simulate_ieee9_hybrid(
            hi, h, nets, delta0, pm, mass, damping)[0]:
        hi *= 1.5
        if hi > 5.0:
            raise RuntimeError("failed to bracket Hybrid EC-SAV CCT")
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if simulate_ieee9_hybrid(
                mid, h, nets, delta0, pm, mass, damping)[0]:
            lo = mid
        else:
            hi = mid
    cct = 0.5 * (lo + hi)
    _, spread, stats = simulate_ieee9_hybrid(
        lo, h, nets, delta0, pm, mass, damping)
    return cct, spread, stats


def ieee9_cct_experiment():
    nets, delta0, pm, _ = case9.make_networks()
    mass = case9.MACHINE_M
    damping = 0.08 * mass
    cct_ref = case9.reference_cct(nets, delta0, pm, mass, damping)
    rows = []
    for h in (0.005, 0.010, 0.020, 0.040):
        cct, spread, stats = hybrid_cct(
            h, nets, delta0, pm, mass, damping, cct_ref)
        rows.append({
            "h_s": h,
            "reference_cct_s": cct_ref,
            "hybrid_cct_s": cct,
            "cct_error_ms": 1000.0 * (cct - cct_ref),
            "stable_side_max_spread_rad": spread,
            **stats,
        })
    return rows


def ieee9_baseline_comparison(hybrid_rows):
    nets, delta0, pm, _ = case9.make_networks()
    mass = case9.MACHINE_M
    damping = 0.08 * mass
    cct_ref = hybrid_rows[0]["reference_cct_s"]
    rows = []
    for hybrid_row in hybrid_rows:
        h = hybrid_row["h_s"]
        rows.append({
            "method": "hybrid-ecsav",
            "h_s": h,
            "cct_s": hybrid_row["hybrid_cct_s"],
            "cct_error_ms": hybrid_row["cct_error_ms"],
        })
        for method in ("RK4", "midpoint", "sav", "ecsav"):
            cct, _ = case9.numerical_cct(
                method, h, nets, delta0, pm, mass, damping, cct_ref)
            rows.append({
                "method": method,
                "h_s": h,
                "cct_s": cct,
                "cct_error_ms": 1000.0 * (cct - cct_ref),
            })
    return rows


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    local_rows = smib_local_order_experiment()
    write_rows("hybrid_ecsav_local_orders.csv", local_rows)
    print("SMIB local-order experiment")
    for row in local_rows:
        print(row)

    convergence_rows = smib_global_convergence_experiment()
    write_rows("hybrid_ecsav_global_convergence.csv", convergence_rows)
    print("\nSMIB global-convergence experiment")
    for row in convergence_rows:
        print(row)

    cct_rows = ieee9_cct_experiment()
    write_rows("hybrid_ecsav_ieee9_cct.csv", cct_rows)
    print("\nDamped IEEE 9-bus CCT experiment")
    for row in cct_rows:
        print(row)

    comparison_rows = ieee9_baseline_comparison(cct_rows)
    write_rows("hybrid_ecsav_ieee9_comparison.csv", comparison_rows)
    print("\nDamped IEEE 9-bus method comparison")
    for row in comparison_rows:
        print(row)


if __name__ == "__main__":
    main()
