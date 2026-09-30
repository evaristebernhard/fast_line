#!/usr/bin/env python3
"""Critical-point tests for metric physical-balance projection.

This script isolates the degeneracy issue in the generic physical-balance
projection idea developed alongside ``minimum_balance_ecsav.py``.

It tests three claims:

1. Quadratic equilibrium (harmonic oscillator + RK4):
      r = O(rho^2),  chi = O(rho^2),  eta_PB = |r|/sqrt(chi) = O(rho).
   Hence chi -> 0 is removable when the balance defect vanishes with the same
   amplitude geometry.

2. Flat quartic critical point E(x)=x^4/4:
      chi = O(rho^6), raw lambda = -r/chi = O(rho^-2),
   while the actual normalized correction length remains O(rho).
   This demonstrates why ``lambda * d`` is a bad coordinate near flat critical
   points and why unit metric-normal + correction length is better.

3. Genuine singularity:
      B(x) = r + x^ell/ell, with grad B(0)=0.
   A negative residual has a correction O(|r|^(1/ell)); a positive residual
   is locally unreachable for even ell.  For a p-th order base method, a
   generic r=O(h^(p+1)) therefore produces only O(h^((p+1)/ell)) correction,
   whereas retaining O(h^(p+1)) correction requires
   r=O(h^(ell*(p+1))).

The script writes CSV diagnostics plus a JSON summary.  It intentionally uses
only NumPy so it can be run independently of the power-system case files.

Example
-------
python physical_balance_degeneracy.py --output-dir output/pb_degeneracy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rk4_step(rhs, y, h):
    """One classical RK4 step for an autonomous ODE."""
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(rhs(y), dtype=float)
    k2 = np.asarray(rhs(y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(rhs(y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(rhs(y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def log_slope(x, y):
    """Least-squares power in y ~ C x^power, ignoring non-positive entries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def harmonic_energy(y, omega0):
    q, v = np.asarray(y, dtype=float)
    return 0.5 * (v * v + (omega0 * q) ** 2)


def harmonic_gradient(y, omega0):
    q, v = np.asarray(y, dtype=float)
    return np.array([omega0 * omega0 * q, v], dtype=float)


def harmonic_experiment(h, amplitudes, omega0=1.7):
    """RK4 near a quadratic equilibrium using the energy metric.

    G = diag(omega0^2, 1), so G^{-1} grad(E) = y and
    chi = ||y||_G^2 = rho^2 exactly at the provisional state.
    """
    G = np.diag([omega0 * omega0, 1.0])
    Ginv = np.diag([1.0 / (omega0 * omega0), 1.0])
    base = np.array([0.8, -0.35], dtype=float)

    def rhs(y):
        return np.array([y[1], -(omega0 * omega0) * y[0]], dtype=float)

    rows = []
    for amp in amplitudes:
        y0 = float(amp) * base
        y_star = rk4_step(rhs, y0, h)
        target = harmonic_energy(y0, omega0)
        energy_star = harmonic_energy(y_star, omega0)
        residual = energy_star - target

        b = harmonic_gradient(y_star, omega0)
        d = Ginv @ b
        chi = float(b @ d)
        sqrt_chi = float(np.sqrt(max(chi, 0.0)))
        rho0 = float(np.sqrt(max(2.0 * target, 0.0)))
        rho_star = float(np.sqrt(max(2.0 * energy_star, 0.0)))

        if sqrt_chi > 0.0:
            unit_normal = d / sqrt_chi
            eta_pb = abs(residual) / sqrt_chi
            raw_lambda = -residual / chi
            # Stable form of rho_target-rho_star; avoids cancellation.
            s_exact = -2.0 * residual / (rho0 + rho_star)
            y_corr = y_star + s_exact * unit_normal
            balance_after = harmonic_energy(y_corr, omega0) - target
            unit_norm_error = float(unit_normal @ G @ unit_normal - 1.0)
        else:
            eta_pb = 0.0
            raw_lambda = 0.0
            s_exact = 0.0
            balance_after = residual
            unit_norm_error = 0.0

        rows.append({
            "amplitude": float(amp),
            "rho": rho0,
            "balance_residual_r": residual,
            "chi": chi,
            "sqrt_chi": sqrt_chi,
            "eta_pb": eta_pb,
            "raw_lambda_linearized": raw_lambda,
            "exact_normalized_correction_s": s_exact,
            "balance_after_projection": balance_after,
            "unit_metric_normal_error": unit_norm_error,
            "abs_r_over_rho2": abs(residual) / max(rho0 * rho0, 1e-300),
            "chi_over_rho2": chi / max(rho0 * rho0, 1e-300),
            "eta_over_rho": eta_pb / max(rho0, 1e-300),
            "abs_s_over_rho": abs(s_exact) / max(rho0, 1e-300),
        })

    # Exact equilibrium must be accepted without invoking a direction.
    rows.append({
        "amplitude": 0.0,
        "rho": 0.0,
        "balance_residual_r": 0.0,
        "chi": 0.0,
        "sqrt_chi": 0.0,
        "eta_pb": 0.0,
        "raw_lambda_linearized": 0.0,
        "exact_normalized_correction_s": 0.0,
        "balance_after_projection": 0.0,
        "unit_metric_normal_error": 0.0,
        "abs_r_over_rho2": 0.0,
        "chi_over_rho2": 0.0,
        "eta_over_rho": 0.0,
        "abs_s_over_rho": 0.0,
    })

    positive = rows[:-1]
    summary = {
        "residual_vs_rho_slope": log_slope(
            [r["rho"] for r in positive],
            [abs(r["balance_residual_r"]) for r in positive],
        ),
        "chi_vs_rho_slope": log_slope(
            [r["rho"] for r in positive], [r["chi"] for r in positive]
        ),
        "eta_vs_rho_slope": log_slope(
            [r["rho"] for r in positive], [r["eta_pb"] for r in positive]
        ),
        "correction_vs_rho_slope": log_slope(
            [r["rho"] for r in positive],
            [abs(r["exact_normalized_correction_s"]) for r in positive],
        ),
        "max_abs_balance_after_projection": float(max(
            abs(r["balance_after_projection"]) for r in rows
        )),
        "max_abs_unit_metric_normal_error": float(max(
            abs(r["unit_metric_normal_error"]) for r in rows
        )),
    }
    return rows, summary


def quartic_energy(x):
    x = float(x)
    return 0.25 * x ** 4


def quartic_flat_experiment(h, amplitudes, base_order=4, defect_coefficient=0.2):
    """Flat critical point: raw lambda diverges but physical correction shrinks.

    We prescribe a compatible target ledger

        E_target = E(x*) - c h^(p+1) x*^4,

    so r = c h^(p+1) x*^4.  This isolates the critical-point geometry without
    tying the result to one particular ODE.
    """
    hp = h ** (base_order + 1)
    if 1.0 - 4.0 * defect_coefficient * hp <= 0.0:
        raise ValueError("h/defect coefficient makes the quartic target negative")
    factor = (1.0 - 4.0 * defect_coefficient * hp) ** 0.25

    rows = []
    for amp in amplitudes:
        x = float(amp)
        rho = abs(x)
        residual = defect_coefficient * hp * x ** 4
        target = quartic_energy(x) - residual
        grad = x ** 3
        chi = grad * grad
        sqrt_chi = abs(grad)

        # The old lambda*d coordinate becomes singular: lambda ~ rho^-2.
        raw_lambda = -residual / chi if chi > 0.0 else 0.0
        eta_pb = abs(residual) / sqrt_chi if sqrt_chi > 0.0 else 0.0

        # Unit normal is sign(x) in the Euclidean metric. Exact local root is
        # x_corr=x*factor and its correction length is O(h^(p+1) rho).
        x_corr = x * factor
        s_exact = x_corr - x
        balance_after = quartic_energy(x_corr) - target

        rows.append({
            "amplitude": x,
            "rho": rho,
            "h": h,
            "base_order_p": base_order,
            "balance_residual_r": residual,
            "chi": chi,
            "sqrt_chi": sqrt_chi,
            "raw_lambda_linearized": raw_lambda,
            "eta_pb": eta_pb,
            "exact_normalized_correction_s": s_exact,
            "balance_after_projection": balance_after,
            "r_over_h_p1_rho4": residual / max(hp * rho ** 4, 1e-300),
            "chi_over_rho6": chi / max(rho ** 6, 1e-300),
            "eta_over_h_p1_rho": eta_pb / max(hp * rho, 1e-300),
            "abs_s_over_h_p1_rho": abs(s_exact) / max(hp * rho, 1e-300),
        })

    rows.append({
        "amplitude": 0.0,
        "rho": 0.0,
        "h": h,
        "base_order_p": base_order,
        "balance_residual_r": 0.0,
        "chi": 0.0,
        "sqrt_chi": 0.0,
        "raw_lambda_linearized": 0.0,
        "eta_pb": 0.0,
        "exact_normalized_correction_s": 0.0,
        "balance_after_projection": 0.0,
        "r_over_h_p1_rho4": 0.0,
        "chi_over_rho6": 0.0,
        "eta_over_h_p1_rho": 0.0,
        "abs_s_over_h_p1_rho": 0.0,
    })

    positive = rows[:-1]
    summary = {
        "residual_vs_rho_slope": log_slope(
            [r["rho"] for r in positive],
            [abs(r["balance_residual_r"]) for r in positive],
        ),
        "chi_vs_rho_slope": log_slope(
            [r["rho"] for r in positive], [r["chi"] for r in positive]
        ),
        "raw_lambda_vs_rho_slope": log_slope(
            [r["rho"] for r in positive],
            [abs(r["raw_lambda_linearized"]) for r in positive],
        ),
        "eta_vs_rho_slope": log_slope(
            [r["rho"] for r in positive], [r["eta_pb"] for r in positive]
        ),
        "correction_vs_rho_slope": log_slope(
            [r["rho"] for r in positive],
            [abs(r["exact_normalized_correction_s"]) for r in positive],
        ),
        "max_abs_balance_after_projection": float(max(
            abs(r["balance_after_projection"]) for r in rows
        )),
    }
    return rows, summary


def singular_barrier_experiment(steps, base_order=4, singular_orders=(2, 4)):
    """Validate the genuine-singularity order barrier.

    At y*=0 use B(x)=r+x^ell/ell with even ell, so grad B(0)=0.
    For r<0 the nearest root has |x|=(-ell*r)^(1/ell). For r>0 there is
    no local real root. Two residual families are compared:

      generic:          r = -h^(p+1)
      order-preserving: r = -h^(ell*(p+1))
    """
    rows = []
    summary = {}

    for ell in singular_orders:
        ell = int(ell)
        if ell < 2 or ell % 2 != 0:
            raise ValueError("singular_orders must contain even integers >= 2")

        for family, exponent in (
            ("generic_residual", base_order + 1),
            ("order_preserving_residual", ell * (base_order + 1)),
        ):
            corr_values = []
            for h in steps:
                residual = -(float(h) ** exponent)
                correction = (-ell * residual) ** (1.0 / ell)
                corr_values.append(correction)
                rows.append({
                    "singular_order_ell": ell,
                    "family": family,
                    "h": float(h),
                    "residual_r": residual,
                    "correction_norm": correction,
                    "residual_exponent": exponent,
                    "expected_correction_order": exponent / ell,
                    "locally_reachable": True,
                })

            observed = log_slope(steps, corr_values)
            summary[f"ell{ell}_{family}_observed_order"] = observed
            summary[f"ell{ell}_{family}_expected_order"] = exponent / ell

        # Positive residual is unreachable because x^ell/ell >= 0.
        rows.append({
            "singular_order_ell": ell,
            "family": "positive_residual_unreachable",
            "h": float(steps[0]),
            "residual_r": float(steps[0]) ** (base_order + 1),
            "correction_norm": float("nan"),
            "residual_exponent": base_order + 1,
            "expected_correction_order": float("nan"),
            "locally_reachable": False,
        })
        summary[f"ell{ell}_positive_residual_locally_reachable"] = False

    return rows, summary


def run_checks(summary):
    checks = []

    def close(name, value, target, tol):
        ok = bool(np.isfinite(value) and abs(value - target) <= tol)
        checks.append({
            "name": name,
            "value": float(value),
            "target": float(target),
            "tolerance": float(tol),
            "passed": ok,
        })

    hsum = summary["harmonic"]
    close("harmonic residual ~ rho^2", hsum["residual_vs_rho_slope"], 2.0, 0.03)
    close("harmonic chi ~ rho^2", hsum["chi_vs_rho_slope"], 2.0, 0.03)
    close("harmonic eta ~ rho", hsum["eta_vs_rho_slope"], 1.0, 0.03)
    close("harmonic exact correction ~ rho", hsum["correction_vs_rho_slope"], 1.0, 0.03)

    qsum = summary["quartic"]
    close("quartic residual ~ rho^4", qsum["residual_vs_rho_slope"], 4.0, 0.03)
    close("quartic chi ~ rho^6", qsum["chi_vs_rho_slope"], 6.0, 0.03)
    close("quartic raw lambda ~ rho^-2", qsum["raw_lambda_vs_rho_slope"], -2.0, 0.03)
    close("quartic eta ~ rho", qsum["eta_vs_rho_slope"], 1.0, 0.03)
    close("quartic exact correction ~ rho", qsum["correction_vs_rho_slope"], 1.0, 0.03)

    ssum = summary["singular_barrier"]
    for key, value in list(ssum.items()):
        if key.endswith("_observed_order"):
            expected_key = key.replace("_observed_order", "_expected_order")
            close(key, value, ssum[expected_key], 1.0e-10)

    return checks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pb_degeneracy"),
        help="directory for CSV/JSON diagnostics",
    )
    parser.add_argument(
        "--h", type=float, default=0.1, help="fixed step for amplitude tests"
    )
    parser.add_argument(
        "--base-order", type=int, default=4, help="formal order p used in scaling tests"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="write diagnostics without failing when expected slopes are not observed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    harmonic_amplitudes = 2.0 ** (-np.arange(0, 13, dtype=float))
    quartic_amplitudes = 10.0 ** (-np.arange(0, 9, dtype=float))
    barrier_steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125], dtype=float)

    harmonic_rows, harmonic_summary = harmonic_experiment(
        args.h, harmonic_amplitudes
    )
    quartic_rows, quartic_summary = quartic_flat_experiment(
        args.h, quartic_amplitudes, base_order=args.base_order
    )
    barrier_rows, barrier_summary = singular_barrier_experiment(
        barrier_steps, base_order=args.base_order, singular_orders=(2, 4)
    )

    summary = {
        "harmonic": harmonic_summary,
        "quartic": quartic_summary,
        "singular_barrier": barrier_summary,
    }
    checks = run_checks(summary)
    summary["checks"] = checks
    summary["all_checks_passed"] = all(c["passed"] for c in checks)

    write_csv(out / "harmonic_amplitude_scaling.csv", harmonic_rows)
    write_csv(out / "quartic_flat_criticality.csv", quartic_rows)
    write_csv(out / "genuine_singularity_order_barrier.csv", barrier_rows)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\nWrote diagnostics to: {out}")

    if not args.no_check and not summary["all_checks_passed"]:
        failed = [c["name"] for c in checks if not c["passed"]]
        raise SystemExit("Scaling checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
