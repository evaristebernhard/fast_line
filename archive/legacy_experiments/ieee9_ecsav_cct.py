"""IEEE 9-bus / 3-machine transient-stability experiment.

This is a deliberately small, reproducible classical-generator benchmark.
The MATPOWER case9 topology and operating-point data are used for the
initial AC power flow.  The transient network is lossless (series R and
constant-P load conductance are omitted) so that the reduced generator
model has the gradient structure required by SAV/EC-SAV:

    M delta_ddot + D delta_dot + grad U(delta) = Pm.

The disturbance is a balanced three-phase fault at bus 7, followed by
tripping line 7--8.  The code compares RK4, implicit midpoint/Newton,
SAV-CN, and EC-SAV-CN.  It also reports admissibility failures of the
energy correction rather than silently falling back to an uncorrected
state.

No third-party power-system package is required; only NumPy and SciPy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


# MATPOWER case9 data (per unit, 100 MVA base).  Bus numbering in this file
# is zero based; generator buses are 1, 2, 3 in the conventional numbering.
BRANCH = np.array([
    # from, to,       R,       X,       B/2 (MATPOWER B is total)
    [1, 4, 0.0000, 0.0576, 0.000],
    [4, 5, 0.0170, 0.0920, 0.079],
    [5, 6, 0.0390, 0.1700, 0.179],
    [3, 6, 0.0000, 0.0586, 0.000],
    [6, 7, 0.0119, 0.1008, 0.1045],
    [7, 8, 0.0085, 0.0720, 0.0745],
    [8, 2, 0.0000, 0.0625, 0.000],
    [8, 9, 0.0320, 0.1610, 0.153],
    [9, 4, 0.0100, 0.0850, 0.088],
], dtype=float)

GEN_BUSES = np.array([0, 1, 2], dtype=int)
LOAD_P = np.array([0.0, 0.0, 0.0, 0.0, 0.90, 0.0, 1.00, 0.0, 1.25])
LOAD_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.30, 0.0, 0.35, 0.0, 0.50])
GEN_P = np.array([0.723, 1.630, 0.850])
GEN_Q = np.array([0.2703, 0.0654, -0.1095])
VSET = np.array([1.040, 1.025, 1.025])

# Classical transient reactances and inertia constants.  M=2H/omega_s;
# omega_s=2*pi*60.  The values are a standard small-signal transient model
# used with the 9-bus benchmark, not a claim that the static case uniquely
# determines machine dynamics.
XDP = np.array([0.0608, 0.1198, 0.1813])
H_INERTIA = np.array([23.64, 6.40, 3.01])
OMEGA_S = 2.0 * np.pi * 60.0
MACHINE_M = 2.0 * H_INERTIA / OMEGA_S


def build_ybus(branches: np.ndarray, *, lossless: bool, open_ids=(),
               shunt_b: np.ndarray | None = None,
               fault_bus: int | None = None, fault_x: float = 1.0e-4) -> np.ndarray:
    """Build a 9-bus Ybus, optionally with a balanced 3-phase shunt fault."""
    ybus = np.zeros((9, 9), dtype=complex)
    open_ids = set(open_ids)
    for k, row in enumerate(branches):
        if k in open_ids:
            continue
        i, j = row[:2].astype(int) - 1
        r, x, bhalf = row[2:]
        z = 1j * x if lossless else r + 1j * x
        y = 1.0 / z
        ybus[i, i] += y + (1j * 2.0 * bhalf if not lossless else 0.0) / 2.0
        ybus[j, j] += y + (1j * 2.0 * bhalf if not lossless else 0.0) / 2.0
        ybus[i, j] -= y
        ybus[j, i] -= y
    if shunt_b is not None:
        ybus[np.diag_indices(9)] += 1j * shunt_b
    if fault_bus is not None:
        # A solid balanced fault is represented by a very small inductive
        # path to ground.  The three phases are identical, hence a positive
        # sequence scalar model is sufficient here.
        ybus[fault_bus, fault_bus] += 1.0 / (1j * fault_x)
    return ybus


def power_injections(V: np.ndarray, ybus: np.ndarray) -> np.ndarray:
    return V * np.conj(ybus @ V)


def solve_case9_power_flow() -> tuple[np.ndarray, np.ndarray]:
    """Solve the static case9 AC power flow with a compact finite-difference Newton method."""
    # bus 1 slack, buses 2--3 PV, buses 4--9 PQ in MATPOWER numbering
    pv = np.array([1, 2], dtype=int)
    pq = np.array([3, 4, 5, 6, 7, 8], dtype=int)
    non_slack = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
    p_spec = -LOAD_P.copy()
    p_spec[GEN_BUSES] = GEN_P
    q_spec = -LOAD_Q.copy()
    q_spec[GEN_BUSES] = GEN_Q

    y = build_ybus(BRANCH, lossless=False)
    theta = np.zeros(9)
    vm = np.ones(9)
    vm[GEN_BUSES] = VSET

    def residual(x: np.ndarray) -> np.ndarray:
        theta[non_slack] = x[:8]
        vm[pq] = x[8:]
        V = vm * np.exp(1j * theta)
        S = power_injections(V, y)
        return np.r_[S.real[non_slack] - p_spec[non_slack],
                     S.imag[pq] - q_spec[pq]]

    x = np.r_[theta[non_slack], vm[pq]]
    for _ in range(30):
        f = residual(x)
        if np.linalg.norm(f, np.inf) < 2.0e-12:
            break
        jac = np.empty((14, 14))
        eps = 2.0e-7
        for j in range(14):
            xp = x.copy()
            xp[j] += eps
            jac[:, j] = (residual(xp) - f) / eps
        x += np.linalg.solve(jac, -f)
    else:
        raise RuntimeError(f"case9 power flow did not converge: {np.linalg.norm(f, np.inf):.3e}")

    theta[non_slack] = x[:8]
    vm[pq] = x[8:]
    V = vm * np.exp(1j * theta)
    return V, power_injections(V, y)


def add_generator_branches(ybus: np.ndarray) -> np.ndarray:
    """Append internal generator emf nodes and connect them through X'_d."""
    yaug = np.zeros((12, 12), dtype=complex)
    yaug[:9, :9] = ybus
    for k, bus in enumerate(GEN_BUSES):
        internal = 9 + k
        y = 1.0 / (1j * XDP[k])
        yaug[bus, bus] += y
        yaug[internal, internal] += y
        yaug[bus, internal] -= y
        yaug[internal, bus] -= y
    return yaug


def kron_reduce_internal(yaug: np.ndarray) -> np.ndarray:
    keep = np.arange(9, 12)
    elim = np.arange(0, 9)
    yii = yaug[np.ix_(keep, keep)]
    yie = yaug[np.ix_(keep, elim)]
    yee = yaug[np.ix_(elim, elim)]
    yei = yaug[np.ix_(elim, keep)]
    return yii - yie @ np.linalg.solve(yee, yei)


@dataclass(frozen=True)
class ReducedNetwork:
    name: str
    yred: np.ndarray
    e_mag: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_k: np.ndarray

    @classmethod
    def from_ybus(cls, name: str, ybus: np.ndarray, e_mag: np.ndarray) -> "ReducedNetwork":
        yred = kron_reduce_internal(add_generator_branches(ybus))
        # For a lossless reciprocal network Yred=jB and B_ij>0 for the
        # off-diagonal generator couplings.  Symmetrize only at roundoff.
        if np.max(np.abs(yred.real)) > 2.0e-9:
            raise ValueError(f"{name}: reduced network is not lossless")
        b = 0.5 * (yred.imag + yred.imag.T)
        ii, jj = np.triu_indices(3, k=1)
        kk = b[ii, jj] * e_mag[ii] * e_mag[jj]
        if np.any(kk <= 0.0):
            raise ValueError(f"{name}: non-positive coupling coefficients {kk}")
        return cls(name, 1j * b, e_mag, ii, jj, kk)

    def potential(self, delta: np.ndarray) -> float:
        phase = delta[self.edge_i] - delta[self.edge_j]
        return float(np.sum(self.edge_k * (1.0 - np.cos(phase))))

    def gradient(self, delta: np.ndarray) -> np.ndarray:
        phase = delta[self.edge_i] - delta[self.edge_j]
        f = self.edge_k * np.sin(phase)
        out = np.zeros(3)
        np.add.at(out, self.edge_i, f)
        np.add.at(out, self.edge_j, -f)
        return out

    def hessian(self, delta: np.ndarray) -> np.ndarray:
        phase = delta[self.edge_i] - delta[self.edge_j]
        out = np.zeros((3, 3))
        for i, j, k, c in zip(self.edge_i, self.edge_j, self.edge_k, np.cos(phase)):
            a = k * c
            out[i, i] += a
            out[j, j] += a
            out[i, j] -= a
            out[j, i] -= a
        return out


def physical_energy(delta, omega, net, mass, damping, pm):
    return (0.5 * np.dot(mass * omega, omega) + net.potential(delta)
            - np.dot(pm, delta))


def make_networks() -> tuple[dict[str, ReducedNetwork], np.ndarray, np.ndarray, np.ndarray]:
    V, S = solve_case9_power_flow()
    Igen = np.conj(S[GEN_BUSES] / V[GEN_BUSES])
    E_internal = V[GEN_BUSES] + 1j * XDP * Igen
    e_mag = np.abs(E_internal)
    delta0 = np.angle(E_internal)

    # Keep the transient model Hamiltonian: reactive loads are retained as
    # shunts, while conductance from constant-P loads is omitted.
    q_shunt = np.zeros(9)
    q_shunt[4] = -LOAD_Q[4] / abs(V[4]) ** 2
    q_shunt[6] = -LOAD_Q[6] / abs(V[6]) ** 2
    q_shunt[8] = -LOAD_Q[8] / abs(V[8]) ** 2

    ypre = build_ybus(BRANCH, lossless=True, shunt_b=q_shunt)
    yfault = build_ybus(BRANCH, lossless=True, shunt_b=q_shunt,
                        fault_bus=6, fault_x=1.0e-4)
    ypost = build_ybus(BRANCH, lossless=True, shunt_b=q_shunt, open_ids=(5,))
    nets = {
        "pre": ReducedNetwork.from_ybus("pre", ypre, e_mag),
        "fault": ReducedNetwork.from_ybus("fault", yfault, e_mag),
        "post": ReducedNetwork.from_ybus("post", ypost, e_mag),
    }
    # Pm is selected from the pre-fault operating point, making the chosen
    # lossless transient model exactly stationary before the disturbance.
    pm = nets["pre"].gradient(delta0)
    damping = np.zeros(3)
    return nets, delta0, pm, damping


def sav_step(delta, omega, r, h, net, mass, damping, pm, C):
    dbar = delta + 0.5 * h * omega
    den = np.sqrt(net.potential(dbar) + C)
    b = net.gradient(dbar) / den
    a0 = 2.0 * mass / h + damping
    rhs = 2.0 * mass * omega / h + pm - r * b
    inv = 1.0 / a0
    z = inv * rhs
    q = inv * b
    alpha = h / 4.0
    v = z - q * (alpha * np.dot(b, z)) / (1.0 + alpha * np.dot(b, q))
    return delta + h * v, 2.0 * v - omega, r + 0.5 * h * np.dot(b, v)


def midpoint_step(delta, omega, h, net, mass, damping, pm):
    omega2 = omega + h * (pm - net.gradient(delta) - damping * omega) / mass
    for _ in range(20):
        dm = delta + 0.25 * h * (omega + omega2)
        wm = 0.5 * (omega + omega2)
        F = mass * (omega2 - omega) / h + damping * wm + net.gradient(dm) - pm
        if np.linalg.norm(F, np.inf) < 2.0e-13:
            break
        J = np.diag(mass / h + damping / 2.0) + (h / 4.0) * net.hessian(dm)
        d_omega = np.linalg.solve(J, -F)
        omega2 += d_omega
        if np.linalg.norm(d_omega, np.inf) < 2.0e-13:
            break
    else:
        raise RuntimeError("implicit midpoint Newton failed")
    return delta + 0.5 * h * (omega + omega2), omega2


def ec_correct(delta, omega_old, omega_tilde, h, net, mass, damping, pm,
               C, Hprev, conserve_momentum=True):
    """Correct one provisional SAV step by solving one scalar quadratic."""
    if conserve_momentum:
        p = np.dot(mass, omega_tilde)
        c_scalar = p / np.sum(mass)
        c = np.full(3, c_scalar)
    else:
        c = np.zeros(3)
    q = omega_tilde - c
    z = omega_old + c
    V = net.potential(delta) - np.dot(pm, delta)
    A = 0.5 * np.dot(mass * q, q) + h / 4.0 * np.dot(damping * q, q)
    B = np.dot(mass * c, q) + h / 2.0 * np.dot(damping * z, q)
    Cq = (0.5 * np.dot(mass * c, c) + V - Hprev
          + h / 4.0 * np.dot(damping * z, z))
    disc = B * B - 4.0 * A * Cq
    if A <= 1.0e-30 or disc < -1.0e-12 * max(1.0, B * B, abs(4.0 * A * Cq)):
        return omega_tilde, False, np.nan, disc
    disc = max(0.0, disc)
    roots = [(-B + np.sqrt(disc)) / (2.0 * A),
             (-B - np.sqrt(disc)) / (2.0 * A)]
    roots = [s for s in roots if np.isfinite(s) and s >= 0.0]
    if not roots:
        return omega_tilde, False, np.nan, disc
    s = min(roots, key=lambda value: abs(value - 1.0))
    return c + s * q, True, s, disc


def rhs_factory(net, mass, damping, pm):
    def rhs(_t, y):
        delta, omega = y[:3], y[3:]
        return np.r_[omega, (pm - net.gradient(delta) - damping * omega) / mass]
    return rhs


def advance(delta, omega, r, duration, h, method, net, mass, damping, pm, C,
            stats=None):
    t = 0.0
    while t < duration - 1.0e-14:
        hh = min(h, duration - t)
        if method == "RK4":
            y = np.r_[delta, omega]
            f = rhs_factory(net, mass, damping, pm)
            k1 = f(0.0, y)
            k2 = f(0.0, y + hh * k1 / 2.0)
            k3 = f(0.0, y + hh * k2 / 2.0)
            k4 = f(0.0, y + hh * k3)
            y = y + hh * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            delta, omega = y[:3], y[3:]
        elif method == "midpoint":
            delta, omega = midpoint_step(delta, omega, hh, net, mass, damping, pm)
        elif method == "sav":
            delta, omega, r = sav_step(delta, omega, r, hh, net, mass, damping, pm, C)
        else:
            raise ValueError(f"advance() does not handle method={method!r}")
        t += hh
    return delta, omega, r


def advance_ec(delta, omega, r, duration, h, net, mass, damping, pm, C, stats):
    t = 0.0
    while t < duration - 1.0e-14:
        hh = min(h, duration - t)
        Hprev = physical_energy(delta, omega, net, mass, damping, pm)
        d2, wt, r2 = sav_step(delta, omega, r, hh, net, mass, damping, pm, C)
        w2, ok, scale, disc = ec_correct(
            d2, omega, wt, hh, net, mass, damping, pm, C, Hprev)
        stats["corrections"] += 1
        stats["min_scale"] = min(stats["min_scale"], scale) if ok else stats["min_scale"]
        stats["max_scale"] = max(stats["max_scale"], scale) if ok else stats["max_scale"]
        stats["max_discriminant"] = max(stats["max_discriminant"], disc)
        if not ok:
            stats["failures"] += 1
            raise RuntimeError(
                f"EC-SAV inadmissible at t={t:.6g}, h={hh:.3g}, discriminant={disc}")
        delta, omega, r = d2, w2, np.sqrt(net.potential(d2) + C)
        t += hh
    return delta, omega, r


def spread(delta):
    return float(np.max(delta) - np.min(delta))


def simulate_fault(tc, h, method, nets, delta0, pm, mass, damping,
                   C=1.0, post_time=2.0):
    delta, omega = delta0.copy(), np.zeros(3)
    net = nets["fault"]
    r = np.sqrt(net.potential(delta) + C)
    stats = {"corrections": 0, "failures": 0, "min_scale": np.inf,
             "max_scale": -np.inf, "max_discriminant": -np.inf}
    if method == "ecsav":
        delta, omega, r = advance_ec(delta, omega, r, tc, h, net, mass, damping, pm, C, stats)
    else:
        delta, omega, r = advance(delta, omega, r, tc, h, method, net, mass, damping, pm, C)

    # Topology change: states are continuous, but H and r are reinitialized
    # with the post-fault potential.  This is the correct event map for the
    # piecewise-Hamiltonian model.
    net = nets["post"]
    r = np.sqrt(net.potential(delta) + C)
    max_spread = spread(delta)
    t = 0.0
    while t < post_time - 1.0e-14:
        hh = min(h, post_time - t)
        if method == "ecsav":
            delta, omega, r = advance_ec(delta, omega, r, hh, hh, net, mass, damping, pm, C, stats)
        else:
            delta, omega, r = advance(delta, omega, r, hh, hh, method, net, mass, damping, pm, C)
        max_spread = max(max_spread, spread(delta))
        if not np.all(np.isfinite(delta)) or max_spread > 20.0 * np.pi:
            break
        t += hh
    return max_spread < np.pi, max_spread, stats


def reference_cct(nets, delta0, pm, mass, damping, lo=0.0, hi=1.0):
    """Bisection CCT from high-accuracy DOP853 and the same spread criterion."""
    def stable(tc):
        y0 = np.r_[delta0, np.zeros(3)]
        sol1 = solve_ivp(rhs_factory(nets["fault"], mass, damping, pm),
                         (0.0, tc), y0, method="DOP853", rtol=2e-10,
                         atol=2e-12, max_step=5e-3)
        sol2 = solve_ivp(rhs_factory(nets["post"], mass, damping, pm),
                         (tc, tc + 2.0), sol1.y[:, -1], method="DOP853",
                         rtol=2e-10, atol=2e-12, max_step=5e-3)
        angle_spread = np.max(np.max(sol2.y[:3], axis=0)
                              - np.min(sol2.y[:3], axis=0))
        return angle_spread < np.pi

    if not stable(lo):
        raise RuntimeError("reference system is unstable even without a fault")
    while stable(hi):
        hi *= 1.5
        if hi > 5.0:
            raise RuntimeError("could not bracket reference CCT")
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def numerical_cct(method, h, nets, delta0, pm, mass, damping, cct_ref):
    lo, hi = 0.0, max(0.5, cct_ref * 2.0)
    while simulate_fault(hi, h, method, nets, delta0, pm, mass, damping)[0]:
        hi *= 1.5
        if hi > 5.0:
            return np.nan, None
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if simulate_fault(mid, h, method, nets, delta0, pm, mass, damping)[0]:
            lo = mid
        else:
            hi = mid
    stable, max_spread, stats = simulate_fault(lo, h, method, nets, delta0, pm, mass, damping)
    return 0.5 * (lo + hi), (max_spread, stats)


def main():
    nets, delta0, pm, damping = make_networks()
    cct_ref = reference_cct(nets, delta0, pm, MACHINE_M, damping)
    print("Reduced coupling coefficients K (pre/fault/post):")
    for name, net in nets.items():
        print(f"  {name:6s}: {net.edge_k}")
    print(f"Initial rotor angles [deg]: {np.degrees(delta0)}")
    print(f"Pm [pu]: {pm}")
    print(f"Reference CCT: {cct_ref:.9f} s")

    rows = []
    for h in (0.005, 0.010, 0.020):
        for method in ("RK4", "midpoint", "sav", "ecsav"):
            cct, extra = numerical_cct(method, h, nets, delta0, pm, MACHINE_M, damping, cct_ref)
            max_spread, stats = extra if extra is not None else (np.nan, {})
            rows.append({
                "method": method,
                "h_s": h,
                "cct_s": cct,
                "cct_error_ms": 1000.0 * (cct - cct_ref),
                "max_spread_rad": max_spread,
                "correction_failures": stats.get("failures", 0),
                "min_scale": stats.get("min_scale", np.nan),
                "max_scale": stats.get("max_scale", np.nan),
            })
            print(rows[-1])

    out = Path("ieee9_cct_ecsav.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
