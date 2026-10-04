"""Kernel (RBF MMD) comparison on the moment-matched pair."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import analysis_common as E                                        # noqa: E402

t0 = time.time()
SMOKE = os.environ.get("E7_SMOKE", "0") == "1"
N_PERM = 400 if not SMOKE else 50
N_MIX = 20_000 if not SMOKE else 1_500
EXACT_CAP = 20_000 if not SMOKE else 2_000
CS = [2.0, 5.0]
BASE = "EDM"

z = np.load(f"{E.P3}/e7a_laws.npz")
tail_idx, bulk_idx, ref_idx = z["tail_idx"], z["bulk_idx"], z["ref_idx"]
tau, P = float(z["tau"]), float(z["p"])
LAWS = ["sparse", "dense", "mom2", "random"]
Ws = {k: z[f"w_{k}"] for k in LAWS}
LOC = {k: z[f"loc_{k}"] for k in LAWS}
print(f"loaded  laws: |Q_T|={len(tail_idx)} pool={len(bulk_idx)}", flush=True)

Ereal = E.real_emb()
Phi_T = np.asarray(Ereal[tail_idx], np.float64)
Phi_pool = np.asarray(Ereal[bulk_idx], np.float64)
nT = len(Phi_T)

# ---------------- median-heuristic bandwidth (fixed, from the reference) ----
rng = np.random.default_rng(31337)
sub = np.sort(rng.choice(ref_idx, size=2000, replace=False))
Xs = np.asarray(Ereal[sub], np.float64)
sq = (Xs ** 2).sum(1)
d2 = sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T
iu = np.triu_indices(len(Xs), 1)
SIGMA = float(np.sqrt(np.median(np.clip(d2[iu], 0, None))))
GAMMA = 1.0 / (2 * SIGMA ** 2)
print(f"median-heuristic sigma = {SIGMA:.5f} (gamma = {GAMMA:.6e})", flush=True)
del d2, Xs


def _rbf_block(X, Y):
    d = ((X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2.0 * (X @ Y.T))
    np.maximum(d, 0.0, out=d)
    np.exp(-GAMMA * d, out=d)
    return d


def gram(X, chunk=4000):
    n = len(X)
    K = np.empty((n, n), np.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        K[a:b] = _rbf_block(X[a:b], X)
    return K


def weighted_mmd2(Xa, wa, Xb, wb, chunk=4000):
    def quad(X, w, Y, v):
        tot = 0.0
        for a in range(0, len(X), chunk):
            b = min(a + chunk, len(X))
            tot += float(w[a:b] @ (_rbf_block(X[a:b], Y) @ v))
        return tot
    return (quad(Xa, wa, Xa, wa) + quad(Xb, wb, Xb, wb)
            - 2 * quad(Xa, wa, Xb, wb))


def mmd_perm_test(X, Y, nperm=N_PERM, seed=0):
    """Unbiased MMD_u^2 + permutation p-value (one Gram, batched K @ D)."""
    n = len(X)
    assert len(Y) == n, "equal sample sizes assumed"
    K = gram(np.concatenate([X, Y]))
    N = 2 * n
    K1 = K.sum(1)
    Q1 = float(K1.sum())
    tr = float(np.trace(K))

    def stats(D):
        Q2 = (D * (K @ D)).sum(0)
        Q3 = K1 @ D
        aKa = ((Q1 + Q2) / 2 + Q3) / 2
        bKb = ((Q1 + Q2) / 2 - Q3) / 2
        aKb = (Q1 - Q2) / 4
        return ((aKa - tr / 2) / (n * (n - 1)) + (bKb - tr / 2) / (n * (n - 1))
                - 2 * aKb / (n * n))

    d0 = np.concatenate([np.ones(n), -np.ones(n)])[:, None]
    obs = float(stats(d0)[0])
    r = np.random.default_rng(seed)
    D = np.empty((N, nperm))
    for j in range(nperm):
        pm = r.permutation(N)
        v = np.empty(N)
        v[pm[:n]] = 1.0
        v[pm[n:]] = -1.0
        D[:, j] = v
    null = stats(D)
    pval = (int((null >= obs).sum()) + 1) / (nperm + 1)
    q95 = float(np.quantile(null, 0.95))
    del K, D
    return obs, pval, q95


rows = []
ra = np.random.default_rng(4242)

# =========================================================================
# (B) component-level POPULATION MMD^2(Q_B, Q_T)
# =========================================================================
print("\n=== (B) component population MMD^2(Q_B, Q_T) ===", flush=True)
wT = np.full(nT, 1.0 / nT)
comp = {}
for k in LAWS:
    w = Ws[k] / Ws[k].sum()
    keep = w > 1e-12
    Xb = Phi_pool[LOC[k]][keep]
    wb = w[keep] / w[keep].sum()
    exact = len(Xb) <= EXACT_CAP
    if not exact:
        s = ra.choice(len(Xb), size=EXACT_CAP, replace=True, p=wb)
        Xb, wb = Xb[s], np.full(EXACT_CAP, 1.0 / EXACT_CAP)
    comp[k] = weighted_mmd2(Xb, wb, Phi_T, wT)
    mu_b = Phi_pool[LOC[k]].T @ w
    gap = float(np.linalg.norm(mu_b - Phi_T.mean(0)))
    rows.append(dict(level="B_component_population", base="", c="", eps="",
                     bulk_law=k, n_per_side="", sigma=SIGMA,
                     mmd2_population=comp[k], mmd2_u_sample="", perm_p="",
                     perm_null_q95="", mean_embed_gap=gap,
                     n_star_for_detection="",
                     note=("exact over %d atoms" % len(Xb)) if exact
                          else ("MC over %d weighted draws" % EXACT_CAP)))
    print(f"  {k:7s}: MMD^2(Q_B,Q_T)={comp[k]:.6e}  (support {len(Xb)}, "
          f"{'exact' if exact else 'MC'})  ||dmu||={gap:.3e} "
          f"({time.time()-t0:.0f}s)", flush=True)

# =========================================================================
# (A) component level, finite n: permutation test at n = |Q_T| per side
# =========================================================================
print("\n=== (A) component permutation test, i.i.d. draws from each law ===",
      flush=True)
for n_side in ([nT, 5000] if not SMOKE else [nT]):
    for k in LAWS:
        w = Ws[k] / Ws[k].sum()
        sel = ra.choice(LOC[k], size=n_side, replace=True, p=w)
        tsel = ra.choice(nT, size=n_side, replace=True)      # i.i.d. from Q_T
        obs, pv, q95 = mmd_perm_test(Phi_pool[sel], Phi_T[tsel], seed=101)
        rows.append(dict(level="A_component_finite_n", base="", c="", eps="",
                         bulk_law=k, n_per_side=n_side, sigma=SIGMA,
                         mmd2_population=comp[k], mmd2_u_sample=obs, perm_p=pv,
                         perm_null_q95=q95, mean_embed_gap="",
                         n_star_for_detection="",
                         note="i.i.d. draws from Q_B vs from Q_T"))
        print(f"  n={n_side:5d} {k:7s}: MMD_u^2={obs:.5e}  perm-p={pv:.5f}  "
              f"null q95={q95:.2e} ({time.time()-t0:.0f}s)", flush=True)

# =========================================================================
# (C) mixture level
# =========================================================================
print(f"\n=== (C) mixture level, permutation test at n={N_MIX}/side ===",
      flush=True)
Eb = E.load_model_emb(BASE).astype(np.float64)
rc = np.random.default_rng(555)
for c in CS:
    eps = c * P
    n_mix = int(round(eps * N_MIX))
    for k in ["mom2", "sparse"]:
        w = Ws[k] / Ws[k].sum()
        b1 = rc.choice(len(Eb), size=N_MIX - n_mix, replace=True)
        b2 = rc.choice(len(Eb), size=N_MIX - n_mix, replace=True)
        s1 = rc.choice(LOC[k], size=n_mix, replace=True, p=w)
        s2 = rc.choice(nT, size=n_mix, replace=True)
        obs, pv, q95 = mmd_perm_test(
            np.concatenate([Eb[b1], Phi_pool[s1]]),
            np.concatenate([Eb[b2], Phi_T[s2]]), seed=202)
        pop = eps ** 2 * comp[k]
        nstar = N_MIX * q95 / pop if pop > 0 else float("inf")
        rows.append(dict(level="C_mixture_population", base=BASE, c=c, eps=eps,
                         bulk_law=k, n_per_side="", sigma=SIGMA,
                         mmd2_population=pop, mmd2_u_sample="", perm_p="",
                         perm_null_q95="", mean_embed_gap="",
                         n_star_for_detection=nstar,
                         note="eps^2 * MMD^2(Q_B,Q_T)"))
        rows.append(dict(level="C_mixture_finite_n", base=BASE, c=c, eps=eps,
                         bulk_law=k, n_per_side=N_MIX, sigma=SIGMA,
                         mmd2_population=pop, mmd2_u_sample=obs, perm_p=pv,
                         perm_null_q95=q95, mean_embed_gap="",
                         n_star_for_detection=nstar,
                         note="Q_1 vs Q_2, independent base draws"))
        print(f"  c={c:g} {k:7s}: population MMD^2={pop:.4e}  sample "
              f"MMD_u^2={obs:.4e}  perm-p={pv:.4f}  null q95={q95:.3e}  "
              f"n* ~ {nstar:.3g}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- controls --------------------------------------------------
print("\n=== controls ===", flush=True)
b1 = rc.choice(len(Eb), size=N_MIX, replace=True)
rsel = np.sort(rc.choice(ref_idx, size=N_MIX, replace=False))
obs, pv, q95 = mmd_perm_test(Eb[b1], np.asarray(Ereal[rsel], np.float64),
                             seed=303)
rows.append(dict(level="control_positive", base=BASE, c="", eps="",
                 bulk_law="", n_per_side=N_MIX, sigma=SIGMA,
                 mmd2_population="", mmd2_u_sample=obs, perm_p=pv,
                 perm_null_q95=q95, mean_embed_gap="",
                 n_star_for_detection="", note="EDM vs real (must reject)"))
print(f"  positive control EDM vs real: MMD_u^2={obs:.4e} p={pv:.5f} "
      f"({time.time()-t0:.0f}s)", flush=True)

eps = CS[-1] * P
n_mix = int(round(eps * N_MIX))
b1 = rc.choice(len(Eb), size=N_MIX - n_mix, replace=True)
b2 = rc.choice(len(Eb), size=N_MIX - n_mix, replace=True)
s1 = rc.choice(nT, size=n_mix, replace=True)
s2 = rc.choice(nT, size=n_mix, replace=True)
obs, pv, q95 = mmd_perm_test(np.concatenate([Eb[b1], Phi_T[s1]]),
                             np.concatenate([Eb[b2], Phi_T[s2]]), seed=404)
rows.append(dict(level="control_null", base=BASE, c=CS[-1], eps=eps,
                 bulk_law="", n_per_side=N_MIX, sigma=SIGMA,
                 mmd2_population=0.0, mmd2_u_sample=obs, perm_p=pv,
                 perm_null_q95=q95, mean_embed_gap="",
                 n_star_for_detection="",
                 note="Q_2 vs Q_2 (must not reject)"))
print(f"  null control Q_2 vs Q_2: MMD_u^2={obs:.4e} p={pv:.5f}", flush=True)

E.writecsv(f"{E.DATA}/prop1_kernel.csv", rows)
print(f"\nE7b DONE ({time.time()-t0:.0f}s)", flush=True)
