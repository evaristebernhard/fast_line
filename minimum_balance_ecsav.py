"""Minimum-balance EC-SAV experiments for Swing dynamics.

This file implements the current unified EC-SAV idea:

1. Run the existing linearly implicit SAV-CN predictor.
2. Measure the physical balance defect epsilon.
3. Construct the COI-compatible Riesz-gradient direction in the metric

       ||dx||_h^2 = h^{-2} ddelta^T M ddelta + domega^T M domega.

4. Solve one scalar nonlinear equation so that the corrected state itself
   satisfies the discrete physical dissipation law exactly.
5. Reset the SAV variable from the corrected physical potential.

The script is intended for local experiments. It keeps the earlier
hybrid_ecsav_experiment.py untouched so old/new schemes can be compared.

Dependencies: numpy, scipy and the repository module ieee9_ecsav_cct.py.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import ieee9_ecsav_cct as case9


def total_potential(delta, net, pm):
    """Full mechanical potential V = U_net - Pm^T delta."""
    return float(net.potential(delta) - np.dot(pm, delta))


def total_gradient(delta, net, pm):
    return np.asarray(net.gradient(delta) - pm, dtype=float)


def physical_energy(delta, omega, net, mass, pm):
    return float(0.5 * np.dot(mass * omega, omega)
                 + total_potential(delta, net, pm))


def project_coi(z, mass, enabled):
    """M-orthogonal projection onto 1^T M z = 0."""
    z = np.asarray(z, dtype=float)
    if not enabled:
        return z.copy()
    return z - np.dot(mass, z) / np.sum(mass)


def use_coi_projection(mass, pm):
    # For a rotationally invariant multimachine system sum(Pm)=0.
    # A single-machine-infinite-bus coordinate has no common-angle null mode.
    return len(mass) > 1 and abs(float(np.sum(pm))) <= 1.0e-11


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
    epsilon: float
    chi_h: float
    sigma_h: float
    lam: float
    correction_norm_h: float
    balance_defect: float
    identity_error: float
    newton_iterations: int
    p_delta_norm: float
    p_omega_norm: float


def sav_predictor(delta_prev, delta, omega, r, h, net, mass, damping, pm, C):
    """Second-order linearly implicit SAV predictor.

    Uses second-order extrapolation when a previous angle is available.
    At an event restart, pass delta_prev=None to use the local midpoint
    predictor delta + h*omega/2.
    """
    if delta_prev is None:
        delta_star = delta + 0.5 * h * omega
    else:
        delta_star = 1.5 * delta - 0.5 * delta_prev

    qstar_sq = total_potential(delta_star, net, pm) + C
    if qstar_sq <= 0.0:
        raise RuntimeError(
            f"SAV square root lost positivity: V+C={qstar_sq:.6e}")

    b = total_gradient(delta_star, net, pm) / np.sqrt(qstar_sq)

    # Rank-one SAV-CN linear system.
    a0 = 2.0 * mass / h + damping
    rhs = 2.0 * mass * omega / h - r * b
    inv_a0 = 1.0 / a0
    y = inv_a0 * rhs
    inv_b = inv_a0 * b
    alpha = h / 4.0
    midpoint_velocity = y - inv_b * (
        alpha * np.dot(b, y)
        / (1.0 + alpha * np.dot(b, inv_b))
    )

    delta_tilde = delta + h * midpoint_velocity
    omega_tilde = 2.0 * midpoint_velocity - omega
    r_tilde = r + 0.5 * h * np.dot(b, midpoint_velocity)

    epsilon = total_potential(delta_tilde, net, pm) + C - r_tilde**2

    modified_old = 0.5 * np.dot(mass * omega, omega) + r**2 - C
    modified_new = (
        0.5 * np.dot(mass * omega_tilde, omega_tilde)
        + r_tilde**2 - C
    )
    modified_balance_defect = (
        modified_new - modified_old
        + h * np.dot(damping * midpoint_velocity, midpoint_velocity)
    )

    return Predictor(
        delta=np.asarray(delta_tilde, dtype=float),
        omega=np.asarray(omega_tilde, dtype=float),
        r=float(r_tilde),
        midpoint_velocity=np.asarray(midpoint_velocity, dtype=float),
        delta_star=np.asarray(delta_star, dtype=float),
        epsilon=float(epsilon),
        modified_balance_defect=float(modified_balance_defect),
    )


def balance_residual(delta_new, omega_new, delta_n, omega_n, h,
                     net, mass, damping, pm):
    H_old = physical_energy(delta_n, omega_n, net, mass, pm)
    midpoint = 0.5 * (omega_n + omega_new)
    return float(
        physical_energy(delta_new, omega_new, net, mass, pm)
        - H_old
        + h * np.dot(damping * midpoint, midpoint)
    )


def dynamic_scale(delta, omega, h, net, mass, pm, coi_enabled):
    """sigma_h^2 = ||omega_rel||_M^2 + h^2 ||grad V||_{M^-1}^2."""
    omega_rel = project_coi(omega, mass, coi_enabled)
    grad = total_gradient(delta, net, pm)
    grad_dir = project_coi(grad / mass, mass, coi_enabled)
    return float(np.sqrt(
        max(0.0,
            np.dot(mass * omega_rel, omega_rel)
            + h * h * np.dot(mass * grad_dir, grad_dir))
    ))


def correction_direction(pred, omega_n, h, net, mass, damping, pm):
    """Riesz gradient of the physical balance functional in the h-metric."""
    coi_enabled = use_coi_projection(mass, pm)
    grad = total_gradient(pred.delta, net, pm)
    midpoint = 0.5 * (omega_n + pred.omega)
    g_omega = mass * pred.omega + h * damping * midpoint

    p_delta = h * h * project_coi(grad / mass, mass, coi_enabled)
    p_omega = project_coi(g_omega / mass, mass, coi_enabled)

    chi = float(np.dot(grad, p_delta) + np.dot(g_omega, p_omega))
    if chi < -1.0e-12 * max(1.0, abs(chi)):
        raise RuntimeError(f"negative chi_h from projection: {chi:.6e}")
    chi = max(0.0, chi)

    sigma_h = dynamic_scale(pred.delta, pred.omega, h, net, mass, pm,
                            coi_enabled)
    p_delta_norm = float(np.sqrt(max(
        0.0, np.dot(mass * p_delta, p_delta)))) / h
    p_omega_norm = float(np.sqrt(max(
        0.0, np.dot(mass * p_omega, p_omega))))

    return p_delta, p_omega, chi, sigma_h, p_delta_norm, p_omega_norm


def minimum_balance_correction(pred, delta_n, omega_n, h, net, mass,
                               damping, pm, C, *,
                               tol=2.0e-13, max_iter=20):
    """Exact scalar balance projection along the minimum-correction direction.

    The direction is frozen at the SAV predictor. Newton then solves

        B_n(delta_tilde + lam*p_delta,
            omega_tilde + lam*p_omega) = 0.

    The initial guess is -epsilon/chi_h. A short backtracking safeguard keeps
    Newton on the local root near zero.
    """
    p_delta, p_omega, chi_h, sigma_h, p_delta_norm, p_omega_norm = (
        correction_direction(pred, omega_n, h, net, mass, damping, pm)
    )

    identity = balance_residual(
        pred.delta, pred.omega, delta_n, omega_n, h,
        net, mass, damping, pm)
    identity_error = float(identity - pred.epsilon)

    scale = max(
        1.0,
        abs(physical_energy(delta_n, omega_n, net, mass, pm)),
        abs(physical_energy(pred.delta, pred.omega, net, mass, pm)),
    )
    abs_tol = tol * scale

    if abs(identity) <= abs_tol:
        lam = 0.0
        iterations = 0
        delta = pred.delta.copy()
        omega = pred.omega.copy()
    else:
        chi_tol = 50.0 * np.finfo(float).eps * scale
        if chi_h <= chi_tol:
            raise RuntimeError(
                "minimum-balance direction is degenerate while the balance "
                f"defect is nonzero: chi_h={chi_h:.3e}, "
                f"epsilon={pred.epsilon:.3e}")

        lam = -identity / chi_h
        iterations = 0

        def eval_f(lam_value):
            d = pred.delta + lam_value * p_delta
            w = pred.omega + lam_value * p_omega
            fval = balance_residual(
                d, w, delta_n, omega_n, h,
                net, mass, damping, pm)
            midpoint = 0.5 * (omega_n + w)
            grad_now = total_gradient(d, net, pm)
            g_now = mass * w + h * damping * midpoint
            deriv = float(
                np.dot(grad_now, p_delta) + np.dot(g_now, p_omega))
            return fval, deriv

        for k in range(max_iter):
            iterations = k + 1
            fval, deriv = eval_f(lam)
            if abs(fval) <= abs_tol:
                break
            if abs(deriv) <= 100.0 * np.finfo(float).eps * max(1.0, chi_h):
                raise RuntimeError(
                    f"scalar balance Newton derivative vanished: {deriv:.3e}")

            step = fval / deriv
            candidate = lam - step

            # Backtrack only if Newton does not reduce the local balance
            # residual. This is cheap because each trial is scalar.
            best = candidate
            best_val, _ = eval_f(best)
            trial_step = step
            for _ in range(12):
                if abs(best_val) <= abs(fval):
                    break
                trial_step *= 0.5
                best = lam - trial_step
                best_val, _ = eval_f(best)
            lam = best
        else:
            fval, _ = eval_f(lam)
            raise RuntimeError(
                f"scalar balance Newton failed after {max_iter} iterations: "
                f"residual={fval:.3e}")

        delta = pred.delta + lam * p_delta
        omega = pred.omega + lam * p_omega

    qsq = total_potential(delta, net, pm) + C
    if qsq <= 0.0:
        raise RuntimeError(
            f"corrected SAV square root lost positivity: V+C={qsq:.6e}")
    r = float(np.sqrt(qsq))

    ddelta = delta - pred.delta
    domega = omega - pred.omega
    correction_norm_h = float(np.sqrt(max(
        0.0,
        np.dot(mass * ddelta, ddelta) / (h * h)
        + np.dot(mass * domega, domega)
    )))
    defect = balance_residual(
        delta, omega, delta_n, omega_n, h,
        net, mass, damping, pm)

    return Correction(
        delta=np.asarray(delta, dtype=float),
        omega=np.asarray(omega, dtype=float),
        r=r,
        epsilon=float(pred.epsilon),
        chi_h=float(chi_h),
        sigma_h=float(sigma_h),
        lam=float(lam),
        correction_norm_h=correction_norm_h,
        balance_defect=float(defect),
        identity_error=identity_error,
        newton_iterations=int(iterations),
        p_delta_norm=float(p_delta_norm),
        p_omega_norm=float(p_omega_norm),
    )


class SMIBPotential:
    def __init__(self, coupling):
        self.coupling = float(coupling)

    def potential(self, delta):
        return float(-self.coupling * np.cos(delta[0]))

    def gradient(self, delta):
        return np.array([self.coupling * np.sin(delta[0])])

    def hessian(self, delta):
        return np.array([[self.coupling * np.cos(delta[0])]])


def observed_orders(values, steps):
    values = np.abs(np.asarray(values, dtype=float))
    steps = np.asarray(steps, dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(1, len(values)):
        if values[i] > 0.0 and values[i - 1] > 0.0:
            out[i] = (
                np.log(values[i - 1] / values[i])
                / np.log(steps[i - 1] / steps[i])
            )
    return out


def backward_previous_state(delta, omega, h, net, mass, damping, pm):
    n = len(delta)

    def rhs(_t, y):
        d = y[:n]
        w = y[n:]
        return np.r_[w, (-damping * w - total_gradient(d, net, pm)) / mass]

    y0 = np.r_[delta, omega]
    sol = solve_ivp(
        rhs, (0.0, -h), y0, method="DOP853",
        rtol=2.0e-13, atol=2.0e-15, max_step=max(h / 20.0, 1.0e-8))
    return sol.y[:n, -1]


def smib_local_order_experiment():
    mass = np.array([1.3])
    damping = np.array([0.2])
    pm = np.array([0.4])
    net = SMIBPotential(coupling=1.1)
    C = 3.0

    steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
    cases = {
        "regular": (np.array([0.8]), np.array([0.4])),
        "turning": (np.array([0.8]), np.array([0.0])),
    }

    rows = []
    for case_name, (delta, omega) in cases.items():
        eps, corr = [], []
        temp = []
        for h in steps:
            delta_prev = backward_previous_state(
                delta, omega, h, net, mass, damping, pm)
            r = np.sqrt(total_potential(delta, net, pm) + C)
            pred = sav_predictor(
                delta_prev, delta, omega, r, h,
                net, mass, damping, pm, C)
            corrected = minimum_balance_correction(
                pred, delta, omega, h,
                net, mass, damping, pm, C)

            eps.append(abs(pred.epsilon))
            corr.append(corrected.correction_norm_h)
            temp.append((h, pred, corrected))

        eps_order = observed_orders(eps, steps)
        corr_order = observed_orders(corr, steps)

        for i, (h, pred, corrected) in enumerate(temp):
            rows.append({
                "case": case_name,
                "h": h,
                "epsilon": abs(pred.epsilon),
                "epsilon_order": eps_order[i],
                "chi_h": corrected.chi_h,
                "sigma_h": corrected.sigma_h,
                "lambda": corrected.lam,
                "correction_norm_h": corrected.correction_norm_h,
                "correction_order": corr_order[i],
                "balance_residual": abs(corrected.balance_defect),
                "identity_error": corrected.identity_error,
                "newton_iterations": corrected.newton_iterations,
                "p_delta_norm": corrected.p_delta_norm,
                "p_omega_norm": corrected.p_omega_norm,
            })
    return rows


def near_equilibrium_amplitude_experiment():
    """Check epsilon=O(rho^2 h^3) and ||dx||_h=O(rho h^3)."""
    mass = np.array([1.3])
    damping = np.array([0.2])
    pm = np.array([0.4])
    coupling = 1.1
    net = SMIBPotential(coupling=coupling)
    C = 3.0
    h = 0.02

    delta_eq = np.array([np.arcsin(pm[0] / coupling)])
    k_eq = coupling * np.cos(delta_eq[0])

    amplitudes = np.array([0.32, 0.16, 0.08, 0.04, 0.02, 0.01])
    rows = []
    eps_values, corr_values = [], []

    for amp in amplitudes:
        delta = delta_eq + np.array([amp])
        omega = np.array([0.35 * amp])
        delta_prev = backward_previous_state(
            delta, omega, h, net, mass, damping, pm)
        r = np.sqrt(total_potential(delta, net, pm) + C)

        pred = sav_predictor(
            delta_prev, delta, omega, r, h,
            net, mass, damping, pm, C)
        corrected = minimum_balance_correction(
            pred, delta, omega, h,
            net, mass, damping, pm, C)

        y = delta - delta_eq
        rho = float(np.sqrt(
            np.dot(mass * omega, omega) + k_eq * y[0] ** 2))
        eps_values.append(abs(pred.epsilon))
        corr_values.append(corrected.correction_norm_h)

        rows.append({
            "amplitude": amp,
            "rho": rho,
            "h": h,
            "epsilon": abs(pred.epsilon),
            "epsilon_over_rho2_h3":
                abs(pred.epsilon) / max(rho * rho * h ** 3, 1e-300),
            "correction_norm_h": corrected.correction_norm_h,
            "correction_over_rho_h3":
                corrected.correction_norm_h / max(rho * h ** 3, 1e-300),
            "chi_h": corrected.chi_h,
            "sigma_h": corrected.sigma_h,
            "lambda": corrected.lam,
            "balance_residual": abs(corrected.balance_defect),
        })

    amp_order_eps = observed_orders(eps_values, amplitudes)
    amp_order_corr = observed_orders(corr_values, amplitudes)
    for i, row in enumerate(rows):
        row["epsilon_amplitude_order"] = amp_order_eps[i]
        row["correction_amplitude_order"] = amp_order_corr[i]
    return rows


def smib_global_convergence_experiment():
    mass = np.array([1.3])
    damping = np.array([0.2])
    pm = np.array([0.4])
    net = SMIBPotential(coupling=1.1)
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
        rhs, (0.0, final_time), np.r_[delta0, omega0],
        method="DOP853", rtol=2.0e-13, atol=2.0e-15,
        max_step=2.0e-4).y[:, -1]

    steps = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
    rows = []

    for method in ("sav", "minimum-balance"):
        errors = []
        temp = []
        for h in steps:
            delta_prev = None
            delta = delta0.copy()
            omega = omega0.copy()
            r = np.sqrt(total_potential(delta, net, pm) + C)
            max_balance = 0.0
            max_corr = 0.0
            max_identity = 0.0
            total_newton = 0

            for _ in range(int(round(final_time / h))):
                old_delta = delta.copy()
                pred = sav_predictor(
                    delta_prev, delta, omega, r, h,
                    net, mass, damping, pm, C)
                if method == "sav":
                    delta, omega, r = pred.delta, pred.omega, pred.r
                else:
                    corrected = minimum_balance_correction(
                        pred, delta, omega, h,
                        net, mass, damping, pm, C)
                    delta, omega, r = (
                        corrected.delta, corrected.omega, corrected.r)
                    max_balance = max(
                        max_balance, abs(corrected.balance_defect))
                    max_corr = max(
                        max_corr, corrected.correction_norm_h)
                    max_identity = max(
                        max_identity, abs(corrected.identity_error))
                    total_newton += corrected.newton_iterations
                delta_prev = old_delta

            error = float(np.linalg.norm(np.r_[delta, omega] - reference))
            errors.append(error)
            temp.append((h, max_balance, max_corr, max_identity, total_newton))

        orders = observed_orders(errors, steps)
        for i, data in enumerate(temp):
            h, max_balance, max_corr, max_identity, total_newton = data
            rows.append({
                "method": method,
                "h": h,
                "state_error": errors[i],
                "observed_order": orders[i],
                "max_balance_residual": max_balance,
                "max_correction_norm_h": max_corr,
                "max_identity_error": max_identity,
                "total_scalar_newton_iterations": total_newton,
            })
    return rows


def init_stats():
    return {
        "steps": 0,
        "max_abs_epsilon": 0.0,
        "max_abs_balance_residual": 0.0,
        "max_abs_identity_error": 0.0,
        "max_correction_norm_h": 0.0,
        "max_abs_lambda": 0.0,
        "min_chi_h": np.inf,
        "max_newton_iterations": 0,
        "switch_jump_defect": 0.0,
    }


def update_stats(stats, pred, corrected):
    stats["steps"] += 1
    stats["max_abs_epsilon"] = max(
        stats["max_abs_epsilon"], abs(pred.epsilon))
    stats["max_abs_balance_residual"] = max(
        stats["max_abs_balance_residual"], abs(corrected.balance_defect))
    stats["max_abs_identity_error"] = max(
        stats["max_abs_identity_error"], abs(corrected.identity_error))
    stats["max_correction_norm_h"] = max(
        stats["max_correction_norm_h"], corrected.correction_norm_h)
    stats["max_abs_lambda"] = max(
        stats["max_abs_lambda"], abs(corrected.lam))
    stats["min_chi_h"] = min(stats["min_chi_h"], corrected.chi_h)
    stats["max_newton_iterations"] = max(
        stats["max_newton_iterations"], corrected.newton_iterations)


def advance_balance(delta_prev, delta, omega, r, duration, h, net,
                    mass, damping, pm, C, stats, *,
                    track_spread=False, stop_spread=np.inf):
    elapsed = 0.0
    max_spread = float(np.max(delta) - np.min(delta))

    while elapsed < duration - 1.0e-14:
        hh = min(h, duration - elapsed)
        old_delta = delta.copy()
        pred = sav_predictor(
            delta_prev, delta, omega, r, hh,
            net, mass, damping, pm, C)
        corrected = minimum_balance_correction(
            pred, delta, omega, hh,
            net, mass, damping, pm, C)
        update_stats(stats, pred, corrected)

        delta_prev = old_delta
        delta, omega, r = corrected.delta, corrected.omega, corrected.r
        elapsed += hh

        if track_spread:
            max_spread = max(
                max_spread, float(np.max(delta) - np.min(delta)))
            if max_spread >= stop_spread:
                break

    return delta_prev, delta, omega, r, max_spread


def simulate_ieee9_balance(tc, h, nets, delta0, pm, mass, damping,
                           C=10.0, post_time=2.0):
    delta = delta0.copy()
    delta_prev = None
    omega = np.zeros_like(delta)
    fault = nets["fault"]
    r = np.sqrt(total_potential(delta, fault, pm) + C)
    stats = init_stats()

    delta_prev, delta, omega, r, fault_spread = advance_balance(
        delta_prev, delta, omega, r, tc, h, fault,
        mass, damping, pm, C, stats,
        track_spread=True, stop_spread=np.pi)

    if fault_spread >= np.pi:
        return False, fault_spread, stats

    Hminus = physical_energy(delta, omega, fault, mass, pm)
    jump_exact = (
        total_potential(delta, nets["post"], pm)
        - total_potential(delta, fault, pm)
    )
    Hplus = physical_energy(delta, omega, nets["post"], mass, pm)
    stats["switch_jump_defect"] = abs((Hplus - Hminus) - jump_exact)

    # State is continuous, SAV variable is reinitialized for the new topology.
    # Restart extrapolation so the predictor does not use cross-event history.
    post = nets["post"]
    delta_prev = None
    r = np.sqrt(total_potential(delta, post, pm) + C)

    delta_prev, delta, omega, r, post_spread = advance_balance(
        delta_prev, delta, omega, r, post_time, h, post,
        mass, damping, pm, C, stats,
        track_spread=True, stop_spread=np.pi)

    max_spread = max(fault_spread, post_spread)
    stable = np.isfinite(max_spread) and max_spread < np.pi
    return stable, max_spread, stats


def minimum_balance_cct(h, nets, delta0, pm, mass, damping, cct_ref):
    lo, hi = 0.0, max(0.5, 2.0 * cct_ref)
    while simulate_ieee9_balance(
            hi, h, nets, delta0, pm, mass, damping)[0]:
        hi *= 1.5
        if hi > 5.0:
            raise RuntimeError("failed to bracket minimum-balance CCT")

    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if simulate_ieee9_balance(
                mid, h, nets, delta0, pm, mass, damping)[0]:
            lo = mid
        else:
            hi = mid

    cct = 0.5 * (lo + hi)
    _, spread, stats = simulate_ieee9_balance(
        lo, h, nets, delta0, pm, mass, damping)
    return cct, spread, stats


def ieee9_cct_experiment():
    nets, delta0, pm, _ = case9.make_networks()
    mass = case9.MACHINE_M
    damping = 0.08 * mass
    cct_ref = case9.reference_cct(
        nets, delta0, pm, mass, damping)

    rows = []
    for h in (0.005, 0.010, 0.020, 0.040):
        cct, spread, stats = minimum_balance_cct(
            h, nets, delta0, pm, mass, damping, cct_ref)
        row = {
            "method": "minimum-balance-ecsav",
            "h_s": h,
            "reference_cct_s": cct_ref,
            "cct_s": cct,
            "cct_error_ms": 1000.0 * (cct - cct_ref),
            "stable_side_max_spread_rad": spread,
            **stats,
        }
        rows.append(row)

    baseline_rows = []
    for row in rows:
        h = row["h_s"]
        baseline_rows.append({
            "method": "minimum-balance-ecsav",
            "h_s": h,
            "reference_cct_s": cct_ref,
            "cct_s": row["cct_s"],
            "cct_error_ms": row["cct_error_ms"],
        })
        for method in ("RK4", "midpoint", "sav", "ecsav"):
            cct, _ = case9.numerical_cct(
                method, h, nets, delta0, pm, mass, damping, cct_ref)
            baseline_rows.append({
                "method": method,
                "h_s": h,
                "reference_cct_s": cct_ref,
                "cct_s": cct,
                "cct_error_ms": 1000.0 * (cct - cct_ref),
            })
    return rows, baseline_rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(title, rows):
    print(f"\n{title}")
    for row in rows:
        print(row)


def run_experiments(experiment, output_dir):
    output_dir = Path(output_dir)

    if experiment in ("local", "all"):
        rows = smib_local_order_experiment()
        write_rows(output_dir / "minimum_balance_local_orders.csv", rows)
        print_rows("SMIB local-order experiment", rows)

    if experiment in ("amplitude", "all"):
        rows = near_equilibrium_amplitude_experiment()
        write_rows(output_dir / "minimum_balance_amplitude.csv", rows)
        print_rows("Near-equilibrium amplitude experiment", rows)

    if experiment in ("convergence", "all"):
        rows = smib_global_convergence_experiment()
        write_rows(output_dir / "minimum_balance_global_convergence.csv", rows)
        print_rows("SMIB global-convergence experiment", rows)

    if experiment in ("ieee9", "all"):
        rows, baseline = ieee9_cct_experiment()
        write_rows(output_dir / "minimum_balance_ieee9_cct.csv", rows)
        write_rows(output_dir / "minimum_balance_ieee9_comparison.csv", baseline)
        print_rows("IEEE 9-bus minimum-balance CCT", rows)
        print_rows("IEEE 9-bus baseline comparison", baseline)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("local", "amplitude", "convergence", "ieee9", "all"),
        default="all",
        help="which experiment suite to run")
    parser.add_argument(
        "--output-dir",
        default="output/minimum_balance_ecsav",
        help="directory for CSV outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()
