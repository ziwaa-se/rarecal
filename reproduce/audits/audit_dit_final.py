"""Audit the two DiT-XL/2 configurations on their M=50,000 sample sets under R1, R2 and R3."""
import os
import sys
import time
import csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
DATA = os.path.expandvars("${PROJECT_ROOT}/paper/data")
os.makedirs(DATA, exist_ok=True)

FINAL = {"dit_cfg15": f"{sc.TFP}/dit_cfg1p5_samples_cache.npz",
         "dit_cfg10": f"{sc.TFP}/dit_cfg1_samples_cache.npz"}
PARTIAL = {"dit_cfg15": f"{sc.P2}/dit_cfg1p5_partial.npz",
        "dit_cfg10": f"{sc.P2}/dit_cfg1_partial.npz"}
NAME = {"dit_cfg15": "DiT-cfg1.5", "dit_cfg10": "DiT-cfg1.0"}
ORCH = {("dit_cfg15", 1e-3): 49, ("dit_cfg15", 1e-4): 3,
        ("dit_cfg10", 1e-3): 32, ("dit_cfg10", 1e-4): 1}

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, N = sc.split_indices(8000)
NH = len(hold_idx)
print(f"N={N} n_h={NH}", flush=True)
b = dict(m=None, V=None, lam=None)
import torch
b = dict(seed=8000, m=torch.tensor(L["m"]), V=torch.tensor(L["V"]),
         lam=torch.tensor(L["lam"]), fit_idx=fit_idx, hold_idx=hold_idx, N=N)
tau1 = {1e-3: float(L["tau3"]), 1e-4: float(L["tau4"])}

rows = []


def quad_ci(rho, lo_w, hi_w, p, nh=NH, z=1.96):
    sig = np.sqrt((1 - p) / (nh * p))
    add = z * sig * rho
    lo = rho - np.sqrt((rho - lo_w) ** 2 + add ** 2)
    hi = rho + np.sqrt((hi_w - rho) ** 2 + add ** 2)
    return lo, hi


def report(slug, func, Rm, tau, p, old_rho, old_k):
    a = sc.audit_rho(Rm, tau, p)
    qlo, qhi = quad_ci(a["rho"], a["lo"], a["hi"], p)
    exp = ORCH.get((slug, p)) if func == "S1" else None
    flag = ""
    if exp is not None and a["k"] != exp:
        flag = f"DISAGREES with orchestrator k={exp} — recompute wins"
    rows.append(dict(config=NAME[slug], functional=func, p=p, M=a["M"],
                     k=a["k"], rho=a["rho"], wilson_lo=a["lo"],
                     wilson_hi=a["hi"], quad_lo=qlo, quad_hi=qhi,
                     verdict=sc.verdict(a["lo"], a["hi"]),
                     earlier_rho=old_rho, earlier_k=old_k,
                     note=flag))
    print(f"{NAME[slug]} {func} p={p:g}: k={a['k']} rho={a['rho']:.3f} "
          f"[{a['lo']:.3f},{a['hi']:.3f}] quad[{qlo:.3f},{qhi:.3f}] "
          f"{rows[-1]['verdict']} (old snap rho={old_rho}, k={old_k}) {flag}",
          flush=True)


# ---------------- R1 ----------------
for slug in ["dit_cfg15", "dit_cfg10"]:
    X = sc.load_cache(FINAL[slug])
    M = len(X)
    # prefix check vs partial cache
    Xs = sc.load_cache(PARTIAL[slug])
    nn = min(len(Xs), M)
    pref_ok = np.array_equal(X[:nn], Xs[:nn])
    print(f"[{slug}] final M={M}; partial prefix identical over {nn}: "
          f"{pref_ok}", flush=True)
    W = sc.project_images(X, b)
    np.savez(f"{sc.P2}/model_w100_{slug}_final.npz", W=W)
    Rm = sc.radius_k(W, 50)
    # reproduction check: partial coords through same path
    Ws = np.load(f"{sc.P2}/model_w100_{slug}.npz")["W"]
    Rs_old = sc.radius_k(Ws, 50)
    d = np.abs(Rm[:nn] - Rs_old[:nn]).max() if pref_ok else np.inf
    print(f"[{slug}] S1 radius reproduction over prefix: max|diff|={d:.2e}",
          flush=True)
    assert d < 1e-8, "S1 projection path changed"
    for p in [1e-3, 1e-4]:
        a_old = sc.audit_rho(Rs_old, tau1[p], p)
        report(slug, "S1", Rm, tau1[p], p, round(a_old["rho"], 4), a_old["k"])
    del X, Xs, W

# ---------------- R3 (rebuild per-position patch PCA, verify, final) ------
X8 = np.load(sc.NPY, mmap_mode="r")
fit_sorted = np.sort(fit_idx)
NPOS, K3 = 16, 32

def patches(Xc):
    n = len(Xc)
    P = Xc.reshape(n, 4, 8, 4, 8, 3).transpose(0, 1, 3, 2, 4, 5)
    return P.reshape(n, NPOS, 192).astype(np.float32)

means = np.zeros((NPOS, 192), dtype=np.float64)
covs = np.zeros((NPOS, 192, 192), dtype=np.float64)
nf = len(fit_sorted)
for a in range(0, nf, 50_000):
    P = patches(np.ascontiguousarray(X8[fit_sorted[a:a + 50_000]]))
    means += P.sum(0)
means /= nf
for a in range(0, nf, 50_000):
    P = patches(np.ascontiguousarray(X8[fit_sorted[a:a + 50_000]]))
    for j in range(NPOS):
        d_ = P[:, j] - means[j]
        covs[j] += d_.T @ d_
covs /= nf
W3 = np.zeros((NPOS, 192, K3), dtype=np.float32)
for j in range(NPOS):
    lam, V = np.linalg.eigh(covs[j])
    lam, V = lam[-K3:][::-1], V[:, -K3:][:, ::-1]
    W3[j] = (V / np.sqrt(lam)).astype(np.float32)
print(f"S3 layer rebuilt ({time.time()-t0:.0f}s)", flush=True)

def r3_of(Xc):
    P = patches(Xc)
    rj = np.empty((len(P), NPOS), dtype=np.float64)
    for j in range(NPOS):
        w = (P[:, j] - means[j].astype(np.float32)) @ W3[j]
        rj[:, j] = np.sqrt((w.astype(np.float64) ** 2).sum(1))
    return rj.max(1)

R3_reals = np.load(f"{sc.P2}/r3_reals.npy")
tau3_ = {p: np.quantile(R3_reals[hold_idx], 1 - p) for p in [1e-3, 1e-4]}
# reproduction gate: partial images -> R3 must match saved r3 arrays
for slug in ["dit_cfg15", "dit_cfg10"]:
    Xs = sc.load_cache(PARTIAL[slug])
    r_new = r3_of(Xs)
    r_old = np.load(f"{sc.P2}/r3_{slug}.npy")
    d = np.abs(r_new - r_old).max()
    print(f"[{slug}] S3 reproduction on partial: max|diff|={d:.2e}", flush=True)
    assert d < 1e-6, "S3 layer rebuild does not reproduce reference radii"
    del Xs
for slug in ["dit_cfg15", "dit_cfg10"]:
    X = sc.load_cache(FINAL[slug])
    r_new = r3_of(X)
    np.save(f"{sc.P2}/r3_{slug}_final.npy", r_new)
    r_old = np.load(f"{sc.P2}/r3_{slug}.npy")
    for p in [1e-3, 1e-4]:
        a_old = sc.audit_rho(r_old, tau3_[p], p)
        report(slug, "S3", r_new, tau3_[p], p, round(a_old["rho"], 4), a_old["k"])
    del X
print(f"S3 final done ({time.time()-t0:.0f}s)", flush=True)

# ---------------- R2 (rebuild embedding layer, verify, final) -------------
K2 = 50
E = np.load(f"{sc.P2}/mae_embed_real.npz")["E"].astype(np.float32)
Ef = E[fit_idx]
m2 = Ef.mean(0, dtype=np.float64)
d_ = Ef - m2.astype(np.float32)
C = (d_.astype(np.float64).T @ d_.astype(np.float64)) / len(Ef)
lam2, V2 = np.linalg.eigh(C)
lam2, V2 = lam2[-K2:][::-1], V2[:, -K2:][:, ::-1]
W2 = (V2 / np.sqrt(lam2)).astype(np.float32)
del d_, C

def r2_of(Emat):
    w = (Emat.astype(np.float32) - m2.astype(np.float32)) @ W2
    return np.sqrt((w.astype(np.float64) ** 2).sum(1))

R2_new = r2_of(E)
R2_old = np.load(f"{sc.P2}/r2_reals.npy")
d = np.abs(R2_new - R2_old).max()
print(f"S2 layer reproduction on reals: max|diff|={d:.2e}", flush=True)
assert d < 1e-6, "S2 layer rebuild does not reproduce reference radii"
del E
tau2_ = {p: np.quantile(R2_old[hold_idx], 1 - p) for p in [1e-3, 1e-4]}
for slug in ["dit_cfg15", "dit_cfg10"]:
    Em = np.load(f"{sc.P2}/mae_embed_{slug}_final.npz")["E"]
    r_new = r2_of(Em)
    np.save(f"{sc.P2}/r2_{slug}_final.npy", r_new)
    r_old = np.load(f"{sc.P2}/r2_{slug}.npy")
    for p in [1e-3, 1e-4]:
        a_old = sc.audit_rho(r_old, tau2_[p], p)
        report(slug, "S2", r_new, tau2_[p], p, round(a_old["rho"], 4), a_old["k"])
    del Em

with open(f"{DATA}/audit_dit_final.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote audit_dit_final.csv ({time.time()-t0:.0f}s)", flush=True)
