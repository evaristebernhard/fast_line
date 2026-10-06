
import numpy as np
import csv
import time
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

try:
    import pandas as pd
except ImportError:  # The numerical prototype itself does not need pandas.
    pd = None

def make_network(N, extra_factor=2, seed=1):
    rng = np.random.default_rng(seed)
    edges = {(min(i,(i+1)%N), max(i,(i+1)%N)) for i in range(N)}
    while len(edges) < N*extra_factor:
        a,b = rng.integers(0,N,2)
        if a != b:
            edges.add((min(a,b),max(a,b)))
    edges = sorted(edges)
    I = np.array([e[0] for e in edges], dtype=int)
    J = np.array([e[1] for e in edges], dtype=int)
    K = rng.uniform(0.8,1.2,len(edges))
    M = rng.uniform(0.8,1.2,N)
    D = np.zeros(N)
    delta_eq = 0.15*np.sin(2*np.pi*np.arange(N)/N) + 0.03*rng.standard_normal(N)
    delta_eq -= delta_eq.mean()
    P = gradU(delta_eq,I,J,K,N)
    return I,J,K,M,D,P,delta_eq

def U_energy(delta,I,J,K):
    th = delta[I]-delta[J]
    return np.sum(K*(1-np.cos(th)))

def gradU(delta,I,J,K,N):
    th = delta[I]-delta[J]
    f = K*np.sin(th)
    g = np.zeros(N)
    np.add.at(g,I,f)
    np.add.at(g,J,-f)
    return g

def Hphys(delta,w,I,J,K,M,P):
    return 0.5*np.dot(M*w,w) + U_energy(delta,I,J,K) - np.dot(P,delta)

def sav_step(delta,w,r,h,I,J,K,M,D,P,C):
    N = len(delta)
    dbar = delta + 0.5*h*w
    Ubar = U_energy(dbar,I,J,K)
    b = gradU(dbar,I,J,K,N) / np.sqrt(Ubar + C)

    a0 = 2*M/h + D
    rhs = 2*M*w/h + P - r*b

    # Sherman-Morrison for diag(a0) + (h/4) b b^T
    inv = 1/a0
    alpha = h/4
    z = inv*rhs
    q = inv*b
    v = z - q*(alpha*np.dot(b,z))/(1 + alpha*np.dot(b,q))

    w2 = 2*v - w
    d2 = delta + h*v
    r2 = r + 0.5*h*np.dot(b,v)
    return d2,w2,r2

def midpoint_step(delta,w,h,I,J,K,M,D,P,maxit=12,tol=1e-11):
    N = len(delta)
    w2 = w + h*(P-gradU(delta,I,J,K,N)-D*w)/M
    for it in range(maxit):
        d2 = delta + 0.5*h*(w+w2)
        dm = 0.5*(delta+d2)
        wm = 0.5*(w+w2)
        g = gradU(dm,I,J,K,N)
        F = M*(w2-w)/h + D*wm + g-P
        if np.linalg.norm(F,np.inf) < tol:
            break
        th = dm[I]-dm[J]
        ew = K*np.cos(th)
        rows = np.r_[I,J,I,J]
        cols = np.r_[I,J,J,I]
        vals = np.r_[ew,ew,-ew,-ew]
        Hess = coo_matrix((vals,(rows,cols)),shape=(N,N)).tocsr()
        Jmat = diags(M/h + D/2) + (h/4)*Hess
        dw = spsolve(Jmat,-F)
        w2 += dw
        if np.linalg.norm(dw,np.inf) < tol:
            break
    d2 = delta + 0.5*h*(w+w2)
    return d2,w2,it+1

def ecsav_conservative_step(delta,w,r,h,I,J,K,M,D,P,C,Htarget,strict=False):
    """One conservative EC-SAV step.

    If ``strict`` is true, an inadmissible projection is raised instead of
    silently returning the provisional velocity.  The latter remains the
    default for backward compatibility with the original prototype.
    """
    d2,w2,_ = sav_step(delta,w,r,h,I,J,K,M,D,P,C)

    # Preserve provisional total angular momentum and project to original energy.
    p = np.dot(M,w2)
    wc = p/M.sum()
    rel = w2-wc

    Tcm = 0.5*M.sum()*wc**2
    V = U_energy(d2,I,J,K)-np.dot(P,d2)
    target_rel = Htarget - V - Tcm
    Trel = 0.5*np.dot(M*rel,rel)

    if Trel <= 1e-30 and abs(target_rel) <= 1e-12:
        return d2,w2,np.sqrt(U_energy(d2,I,J,K)+C),True
    if target_rel < 0 or Trel <= 1e-30:
        if strict:
            raise ValueError(
                "EC-SAV conservative projection is inadmissible: "
                f"target_rel={target_rel:.3e}, Trel={Trel:.3e}")
        return d2,w2,np.sqrt(U_energy(d2,I,J,K)+C),False

    s = np.sqrt(target_rel/Trel)
    w2 = wc + s*rel
    r2 = np.sqrt(U_energy(d2,I,J,K)+C)
    return d2,w2,r2,True

def ecsav_damped_step(delta,w,r,h,I,J,K,M,D,P,C,Hprev,strict=False):
    """One damped EC-SAV step with an explicit admissibility diagnostic."""
    d2,wtilde,_ = sav_step(delta,w,r,h,I,J,K,M,D,P,C)

    # Keep the provisional total angular momentum, but enforce
    # H^{n+1}-H^n = -h * ((w^n+w^{n+1})/2)^T D ((w^n+w^{n+1})/2).
    p = np.dot(M,wtilde)
    c = np.full_like(wtilde,p/M.sum())
    q = wtilde-c
    V = U_energy(d2,I,J,K)-np.dot(P,d2)
    z = w+c

    A = 0.5*np.dot(M*q,q) + h/4*np.dot(D*q,q)
    B = np.dot(M*c,q) + h/2*np.dot(D*z,q)
    Cq = 0.5*np.dot(M*c,c) + V - Hprev + h/4*np.dot(D*z,z)

    disc = B*B - 4*A*Cq
    scale = max(1.0, B*B, abs(4*A*Cq))
    if A <= 1e-30 and abs(Cq) <= 1e-12:
        return d2,wtilde,np.sqrt(U_energy(d2,I,J,K)+C),True,0.0
    if A <= 1e-30 or disc < -1e-12*scale:
        if strict:
            raise ValueError(
                "EC-SAV damped projection is inadmissible: "
                f"A={A:.3e}, discriminant={disc:.3e}")
        return d2,wtilde,np.sqrt(U_energy(d2,I,J,K)+C),False
    disc = max(0.0, disc)

    roots = [(-B+np.sqrt(disc))/(2*A),(-B-np.sqrt(disc))/(2*A)]
    roots = [x for x in roots if np.isfinite(x) and x >= 0]
    if not roots:
        if strict:
            raise ValueError(
                "EC-SAV damped projection has no nonnegative root: "
                f"discriminant={disc:.3e}")
        return d2,wtilde,np.sqrt(U_energy(d2,I,J,K)+C),False

    s = min(roots,key=lambda x:abs(x-1))
    w2 = c+s*q
    r2 = np.sqrt(U_energy(d2,I,J,K)+C)
    return d2,w2,r2,True

def benchmark_scaling(sizes=(10,50,100,200,500,1000), steps=100, h=0.02):
    rows = []
    for N in sizes:
        I,J,K,M,D,P,deq = make_network(N,seed=5)
        rng = np.random.default_rng(105)
        d0 = deq + 0.05*rng.standard_normal(N)
        d0 -= d0.mean()
        w0 = 0.03*rng.standard_normal(N)
        w0 -= np.dot(M,w0)/M.sum()
        C = 1.0

        for method in ("midpoint","sav","ecsav"):
            d,w = d0.copy(),w0.copy()
            r = np.sqrt(U_energy(d,I,J,K)+C)
            H0 = Hphys(d,w,I,J,K,M,P)
            iters = 0
            t0 = time.perf_counter()

            for _ in range(steps):
                if method == "midpoint":
                    d,w,it = midpoint_step(d,w,h,I,J,K,M,D,P)
                    iters += it
                elif method == "sav":
                    d,w,r = sav_step(d,w,r,h,I,J,K,M,D,P,C)
                else:
                    d,w,r,ok = ecsav_conservative_step(d,w,r,h,I,J,K,M,D,P,C,H0)
                    if not ok:
                        raise RuntimeError("EC-SAV projection failed")

            elapsed = time.perf_counter()-t0
            rows.append({
                "machines":N,
                "method":method,
                "ms_per_step":1000*elapsed/steps,
                "avg_newton_iters":iters/steps if method=="midpoint" else 0,
                "end_energy_error":abs(Hphys(d,w,I,J,K,M,P)-H0)
            })

    return pd.DataFrame(rows) if pd is not None else rows

if __name__ == "__main__":
    df = benchmark_scaling()
    if pd is not None:
        print(df.to_string(index=False))
        df.to_csv("multimachine_scaling_reproduced.csv",index=False)
    else:
        print(*df, sep="\n")
        with open("multimachine_scaling_reproduced.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=df[0].keys())
            writer.writeheader()
            writer.writerows(df)
