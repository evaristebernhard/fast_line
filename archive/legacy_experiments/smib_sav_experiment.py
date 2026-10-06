import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M = 0.05
D = 0.0
Pm = 0.8
K_pre = 1.2
K_fault = 0.2
K_post = 1.0
C = 1.0

delta0 = np.arcsin(Pm / K_pre)
omega0 = 0.0
delta_s = np.arcsin(Pm / K_post)
delta_u = np.pi - delta_s

def energy(delta, omega, K):
    return 0.5*M*omega**2 + K*(1-np.cos(delta)) - Pm*delta

Hcrit = energy(delta_u, 0.0, K_post)

def rhs(y, K):
    delta, omega = y
    return np.array([omega, (Pm - K*np.sin(delta) - D*omega)/M])

def rhs_fault(t, y):
    return rhs(y, K_fault)

def critical_event(t, y):
    return energy(y[0], y[1], K_post) - Hcrit

critical_event.terminal = True
critical_event.direction = 1

ref = solve_ivp(
    rhs_fault, (0, 2), [delta0, omega0],
    events=critical_event, rtol=1e-12, atol=1e-14,
    method="DOP853", max_step=1e-4
)
t_ref = float(ref.t_events[0][0])

def step_rk4(y, h, K):
    k1 = rhs(y, K)
    k2 = rhs(y+h*k1/2, K)
    k3 = rhs(y+h*k2/2, K)
    k4 = rhs(y+h*k3, K)
    return y+h*(k1+2*k2+2*k3+k4)/6

def newton_step(y, h, K, kind):
    delta, omega = y
    z = y + h*rhs(y, K)
    for _ in range(12):
        d2, w2 = z
        if kind == "trap":
            F1 = d2-delta-h*0.5*(omega+w2)
            F2 = w2-omega-h/(2*M)*(
                (Pm-K*np.sin(delta)-D*omega) +
                (Pm-K*np.sin(d2)-D*w2)
            )
            c = np.cos(d2)
        else:
            dm = 0.5*(delta+d2)
            wm = 0.5*(omega+w2)
            F1 = d2-delta-h*wm
            F2 = w2-omega-h/M*(Pm-K*np.sin(dm)-D*wm)
            c = np.cos(dm)
        J = np.array([
            [1.0, -h/2],
            [h*K*c/(2*M), 1+h*D/(2*M)]
        ])
        dz = np.linalg.solve(J, -np.array([F1, F2]))
        z += dz
        if np.linalg.norm(dz, np.inf) < 1e-13:
            break
    return z

def step_trapezoidal(y, h, K):
    return newton_step(y, h, K, "trap")

def step_midpoint(y, h, K):
    return newton_step(y, h, K, "mid")

def bfun(delta, K):
    U = K*(1-np.cos(delta))
    return K*np.sin(delta) / np.sqrt(U+C)

def step_sav_cn(state, h, K):
    delta, omega, r = state
    delta_bar = delta + 0.5*h*omega
    b = bfun(delta_bar, K)
    denom = 2*M/h + D + h*b*b/4
    v = (2*M*omega/h + Pm - r*b) / denom
    omega2 = 2*v-omega
    delta2 = delta+h*v
    r2 = r+0.5*h*b*v
    return np.array([delta2, omega2, r2])

methods = ["RK4", "Trapezoidal", "Implicit midpoint", "SAV-CN"]

def numerical_cct(method, h):
    y = np.array([delta0, omega0], dtype=float)
    if method == "SAV-CN":
        r0 = np.sqrt(K_fault*(1-np.cos(delta0))+C)
        s = np.array([delta0, omega0, r0], dtype=float)

    g_prev = energy(delta0, omega0, K_post)-Hcrit
    t = 0.0
    for _ in range(10000):
        if method == "RK4":
            y2 = step_rk4(y, h, K_fault)
            d, w = y2
        elif method == "Trapezoidal":
            y2 = step_trapezoidal(y, h, K_fault)
            d, w = y2
        elif method == "Implicit midpoint":
            y2 = step_midpoint(y, h, K_fault)
            d, w = y2
        else:
            s2 = step_sav_cn(s, h, K_fault)
            d, w = s2[:2]

        g = energy(d, w, K_post)-Hcrit
        if g >= 0:
            frac = -g_prev/(g-g_prev)
            return t + frac*h

        g_prev = g
        t += h
        if method == "SAV-CN":
            s = s2
        else:
            y = y2
    return np.nan

rows = []
for h in [0.001, 0.005, 0.010, 0.020, 0.050]:
    for method in methods:
        cct = numerical_cct(method, h)
        rows.append({
            "Method": method,
            "dt_ms": h*1000,
            "CCT_s": cct,
            "CCT_error_ms": (cct-t_ref)*1000,
            "Relative_error_pct": abs(cct-t_ref)/t_ref*100
        })

cct_df = pd.DataFrame(rows)
cct_df.to_csv("cct_comparison.csv", index=False)

tc_test = 0.18
clear = solve_ivp(
    rhs_fault, (0, tc_test), [delta0, omega0],
    rtol=1e-12, atol=1e-14, method="DOP853", max_step=1e-4
)
yc = clear.y[:, -1]

def simulate_post(method, h, T=10.0):
    n = int(round(T/h))
    if method == "SAV-CN":
        r0 = np.sqrt(K_post*(1-np.cos(yc[0]))+C)
        s = np.array([yc[0], yc[1], r0], dtype=float)
        phys = [energy(s[0], s[1], K_post)]
        mod = [0.5*M*s[1]**2+s[2]**2-C-Pm*s[0]]
        for _ in range(n):
            s = step_sav_cn(s, h, K_post)
            phys.append(energy(s[0], s[1], K_post))
            mod.append(0.5*M*s[1]**2+s[2]**2-C-Pm*s[0])
        return np.asarray(phys), np.asarray(mod)

    y = yc.copy()
    phys = [energy(y[0], y[1], K_post)]
    for _ in range(n):
        if method == "RK4":
            y = step_rk4(y, h, K_post)
        elif method == "Trapezoidal":
            y = step_trapezoidal(y, h, K_post)
        else:
            y = step_midpoint(y, h, K_post)
        phys.append(energy(y[0], y[1], K_post))
    return np.asarray(phys), None

erows = []
for h in [0.005, 0.010, 0.020, 0.050]:
    for method in methods:
        phys, mod = simulate_post(method, h, 10.0)
        erows.append({
            "Method": method,
            "dt_ms": h*1000,
            "Max_physical_energy_drift": np.max(np.abs(phys-phys[0])),
            "End_physical_energy_drift": phys[-1]-phys[0],
            "Max_modified_energy_drift":
                np.nan if mod is None else np.max(np.abs(mod-mod[0]))
        })

energy_df = pd.DataFrame(erows)
energy_df.to_csv("energy_drift.csv", index=False)

print(f"Reference CCT = {t_ref:.9f} s")
print(cct_df.to_string(index=False))
print()
print(energy_df.to_string(index=False))
