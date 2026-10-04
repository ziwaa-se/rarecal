"""Fit R3 (maximum of per-patch whitened radii) and audit every configuration under it."""
import os
import sys
import time
import numpy as np
import csv
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
K3 = 32
NPOS = 16

X8 = np.load(sc.NPY, mmap_mode="r")
fit_idx, hold_idx, N = sc.split_indices(8000)
fit_sorted = np.sort(fit_idx)

def patches(Xc):
    """(n,32,32,3) uint8 -> (n,16,192) float32."""
    n = len(Xc)
    P = Xc.reshape(n, 4, 8, 4, 8, 3).transpose(0, 1, 3, 2, 4, 5)
    return P.reshape(n, NPOS, 192).astype(np.float32)

# ---- per-position PCA on the fit split ----
means = np.zeros((NPOS, 192), dtype=np.float64)
covs = np.zeros((NPOS, 192, 192), dtype=np.float64)
nf = len(fit_sorted)
for a in range(0, nf, 50_000):
    P = patches(np.ascontiguousarray(X8[fit_sorted[a:a + 50_000]]))
    means += P.sum(0)
covs_n = 0
means /= nf
for a in range(0, nf, 50_000):
    P = patches(np.ascontiguousarray(X8[fit_sorted[a:a + 50_000]]))
    for j in range(NPOS):
        d = P[:, j] - means[j]
        covs[j] += d.T @ d
covs /= nf
W3 = np.zeros((NPOS, 192, K3), dtype=np.float32)
for j in range(NPOS):
    lam, V = np.linalg.eigh(covs[j])
    lam, V = lam[-K3:][::-1], V[:, -K3:][:, ::-1]  # descending eigenpairs (cols)
    W3[j] = (V / np.sqrt(lam)).astype(np.float32)
P0 = patches(np.ascontiguousarray(X8[fit_sorted[:50_000]]))
w0 = (P0[:, 0] - means[0].astype(np.float32)) @ W3[0]
vv = w0.var(0)
assert np.abs(vv - 1).max() < 0.05, f"whitening broken: {vv[:5]}"
print(f"per-position PCA built; whiten-var range "
      f"[{vv.min():.3f},{vv.max():.3f}] ({time.time()-t0:.0f}s)", flush=True)

def r3_and_argmax(Xc):
    P = patches(Xc)
    rj = np.empty((len(P), NPOS), dtype=np.float64)
    for j in range(NPOS):
        w = (P[:, j] - means[j].astype(np.float32)) @ W3[j]
        rj[:, j] = np.sqrt((w.astype(np.float64) ** 2).sum(1))
    return rj.max(1), rj.argmax(1), rj

# ---- reals ----
R3 = np.empty(N, dtype=np.float64)
AM = np.empty(N, dtype=np.int8)
for a in range(0, N, 100_000):
    r, am, _ = r3_and_argmax(np.ascontiguousarray(X8[a:a + 100_000]))
    R3[a:a + 100_000] = r
    AM[a:a + 100_000] = am
np.save(f"{sc.P2}/r3_reals.npy", R3)
print(f"reals done ({time.time()-t0:.0f}s)", flush=True)

R3f, R3h = R3[fit_idx], R3[hold_idx]

# dominance diagnostic on fit-split top 1%
top = R3f >= np.quantile(R3f, 0.99)
hist = np.bincount(AM[fit_idx][top], minlength=NPOS) / top.sum()
dominant = hist.max()
print(f"argmax histogram (fit top1%): max share={dominant:.2f} "
      f"{np.round(hist, 3).tolist()}", flush=True)
use_pit = dominant > 0.5

# GPD operability
q, u, xi, sg, ksp = sc.select_u(R3f)
tau = {p: np.quantile(R3h, 1 - p) for p in [1e-2, 3e-3, 1e-3, 1e-4]}
oper = [f"S3 GPD adaptive: q={q} xi={xi:+.3f} sg={sg:.3f} KS-p={ksp:.3f} "
        f"(operability-i pass={ksp >= 0.05})",
        f"argmax dominance: max share={dominant:.2f} PIT fallback "
        f"triggered={use_pit}"]

# self-audit (fit split in model role)
for p in [1e-2, 3e-3, 1e-3]:
    a = sc.audit_rho(R3f, tau[p], p)
    oper.append(f"self-audit p={p:g}: rho={a['rho']:.3f} "
                f"[{a['lo']:.3f},{a['hi']:.3f}] covers1="
                f"{a['lo'] <= 1 <= a['hi']}")
    print(oper[-1], flush=True)

# ---- model leaderboard ----
rows = []
cachemap = dict(sc.CONFIGS)
for name, path in sc.CONFIGS:
    X = sc.load_cache(path)
    r, am, _ = r3_and_argmax(X)
    np.save(f"{sc.P2}/r3_{sc.SLUG[name]}.npy", r)
    for p in [1e-2, 3e-3, 1e-3, 1e-4]:
        a = sc.audit_rho(r, tau[p], p)
        rows.append(dict(config=name, p=p, M=a["M"], k=a["k"], rho=a["rho"],
                         lo=a["lo"], hi=a["hi"],
                         verdict=sc.verdict(a["lo"], a["hi"])))
    r3v = next(r_ for r_ in rows if r_["config"] == name and r_["p"] == 1e-3)
    print(f"{name:12s} rho_S3(1e-3)={r3v['rho']:.3f} "
          f"[{r3v['lo']:.3f},{r3v['hi']:.3f}] {r3v['verdict']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del X

with open(f"{OUT}/s3_leaderboard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/s3_operability.txt", "w") as f:
    f.write("\n".join(oper) + "\n")
print("wrote s3_leaderboard.csv, s3_operability.txt", flush=True)
