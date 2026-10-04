"""Fit R2 (whitened radius on MAE embeddings) and audit every configuration under it."""
import os
import sys
import time
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
K2 = 50

E = np.load(f"{sc.P2}/mae_embed_real.npz")["E"].astype(np.float32)
fit_idx, hold_idx, N = sc.split_indices(8000)
Ef = E[fit_idx]
m2 = Ef.mean(0, dtype=np.float64)
d = Ef - m2.astype(np.float32)
C = (d.astype(np.float64).T @ d.astype(np.float64)) / len(Ef)
lam, V = np.linalg.eigh(C)
lam, V = lam[-K2:][::-1], V[:, -K2:][:, ::-1]   # descending eigenpairs (columns)
W2 = (V / np.sqrt(lam)).astype(np.float32)
wf = (Ef - m2.astype(np.float32)) @ W2
vv = wf.var(0)
assert np.abs(vv - 1).max() < 0.02, f"whitening broken: {vv[:5]}"
print(f"S2 layer built: top-{K2} evr={lam.sum()/np.trace(C):.3f} "
      f"whiten-var range [{vv.min():.3f},{vv.max():.3f}] "
      f"({time.time()-t0:.0f}s)", flush=True)

def r2(Emat):
    w = (Emat.astype(np.float32) - m2.astype(np.float32)) @ W2
    return np.sqrt((w.astype(np.float64) ** 2).sum(1))

R2 = r2(E)
np.save(f"{sc.P2}/r2_reals.npy", R2)
R2f, R2h = R2[fit_idx], R2[hold_idx]
q, u, xi, sg, ksp = sc.select_u(R2f)
tau = {p: np.quantile(R2h, 1 - p) for p in [1e-2, 3e-3, 1e-3, 1e-4]}
oper = [f"S2 GPD adaptive: q={q} xi={xi:+.3f} sg={sg:.3f} KS-p={ksp:.3f} "
        f"(operability-i pass={ksp >= 0.05})"]
for p in [1e-2, 3e-3, 1e-3]:
    a = sc.audit_rho(R2f, tau[p], p)
    oper.append(f"self-audit p={p:g}: rho={a['rho']:.3f} "
                f"[{a['lo']:.3f},{a['hi']:.3f}] covers1="
                f"{a['lo'] <= 1 <= a['hi']}")
    print(oper[-1], flush=True)

rows = []
for name, _ in sc.CONFIGS:
    Em = np.load(f"{sc.P2}/mae_embed_{sc.SLUG[name]}.npz")["E"]
    r = r2(Em)
    np.save(f"{sc.P2}/r2_{sc.SLUG[name]}.npy", r)
    for p in [1e-2, 3e-3, 1e-3, 1e-4]:
        a = sc.audit_rho(r, tau[p], p)
        rows.append(dict(config=name, p=p, M=a["M"], k=a["k"], rho=a["rho"],
                         lo=a["lo"], hi=a["hi"],
                         verdict=sc.verdict(a["lo"], a["hi"])))
    rv = next(r_ for r_ in rows if r_["config"] == name and r_["p"] == 1e-3)
    print(f"{name:12s} rho_S2(1e-3)={rv['rho']:.3f} "
          f"[{rv['lo']:.3f},{rv['hi']:.3f}] {rv['verdict']}", flush=True)

with open(f"{OUT}/s2_leaderboard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/s2_operability.txt", "w") as f:
    f.write("\n".join(oper) + "\n")
print("wrote s2_leaderboard.csv, s2_operability.txt", flush=True)
