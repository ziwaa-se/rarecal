"""Helpers shared by the analysis scripts: loading cached samples, MAE ViT-B/16 features,
Frechet distances and moment-matched mixtures."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import severity as sc                                       # noqa: E402

P2 = sc.P2
P3 = os.path.expandvars("${CACHE_ROOT}/paper")
ROOT = os.path.expandvars("${PROJECT_ROOT}/paper")
DATA = f"{ROOT}/data"
FIGS = f"{ROOT}/figures"

# DiT is audited from the FINAL 50k caches (results_tables.py:40,46-48)
FINAL_SLUG = {"DiT-cfg1.5", "DiT-cfg1.0"}
CONFIG_NAMES = [n for n, _ in sc.CONFIGS]

REAL_SPLIT_SEED = 20260816          # cut of the frozen hold split into REF/POOL
TAU_P = 1e-3                        # audit depth used throughout 


# --------------------------------------------------------------------------
# asset loaders
# --------------------------------------------------------------------------
def _suf(name):
    return "_final" if name in FINAL_SLUG else ""


def emb_path(name):
    return f"{P2}/mae_embed_{sc.SLUG[name]}{_suf(name)}.npz"


def w100_path(name):
    return f"{P2}/model_w100_{sc.SLUG[name]}{_suf(name)}.npz"


def load_model_emb(name):
    """(M, 768) float32 MAE embeddings of a config's audited sample set."""
    return np.load(emb_path(name))["E"].astype(np.float32)


def load_model_r50(name):
    """(M,) R1 severity radii of the same sample set, same row order."""
    return sc.radius_k(np.load(w100_path(name))["W"], 50)


_LAYER = None


def layer():
    global _LAYER
    if _LAYER is None:
        _LAYER = np.load(f"{P2}/layer_seed8000.npz")
    return _LAYER


def tau3():
    return float(layer()["tau3"])


def real_r50():
    return layer()["R50"]


_REAL_E = None


def real_emb():
    """(1281167, 768) float16 -- kept in fp16 to bound RSS; cast per chunk."""
    global _REAL_E
    if _REAL_E is None:
        t0 = time.time()
        _REAL_E = np.load(f"{P2}/mae_embed_real.npz")["E"]
        print(f"  [real MAE embeddings loaded {_REAL_E.shape} "
              f"({time.time()-t0:.0f}s)]", flush=True)
    return _REAL_E


def ref_pool_split():
    """Disjoint halves of the frozen seed-8000 hold split (sorted indices)."""
    _, hold_idx, _ = sc.split_indices(8000)
    perm = np.random.default_rng(REAL_SPLIT_SEED).permutation(len(hold_idx))
    h = hold_idx[perm]
    half = len(h) // 2
    return np.sort(h[:half]), np.sort(h[half:])


# --------------------------------------------------------------------------
# moments and the Frechet distance (FMD)
# --------------------------------------------------------------------------
def moments(X, w=None, chunk=50_000):
    """(mu, S) of a discrete law over rows of X.

 X may be float16/float32; accumulation is float64. w are non-negative
 weights summing to 1 (uniform if None). S is the UNCENTRED second moment
 E[psi psi^T], so (mu, S) is exactly the FID moment map's expectation.
 """
    n, k = X.shape
    mu = np.zeros(k, np.float64)
    S = np.zeros((k, k), np.float64)
    for a in range(0, n, chunk):
        xb = np.asarray(X[a:a + chunk], dtype=np.float64)
        if w is None:
            mu += xb.sum(0)
            S += xb.T @ xb
        else:
            wb = w[a:a + chunk]
            mu += wb @ xb
            S += (xb * wb[:, None]).T @ xb
    if w is None:
        mu /= n
        S /= n
    return mu, S


def cov_of(mu, S):
    return S - np.outer(mu, mu)


def mix_moments(mu0, S0, mu1, S1, eps):
    """Moments of (1-eps) Q_0 + eps Q_1 (moments are linear in the law)."""
    return (1 - eps) * mu0 + eps * mu1, (1 - eps) * S0 + eps * S1


def frechet(mu1, C1, mu2, C2):
    """Frechet (Wasserstein-2 between Gaussians) distance -- the FID formula.

 ||mu1-mu2||^2 + tr + tr - 2 tr((^{1/2} ^{1/2})^{1/2}),
 evaluated by symmetric eigendecomposition (no scipy.sqrtm branch cuts).
 """
    d = np.asarray(mu1, np.float64) - np.asarray(mu2, np.float64)
    C1 = np.asarray(C1, np.float64)
    C2 = np.asarray(C2, np.float64)
    ev, U = np.linalg.eigh((C1 + C1.T) / 2)
    A = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
    Mx = A @ ((C2 + C2.T) / 2) @ A
    ev2 = np.linalg.eigvalsh((Mx + Mx.T) / 2)
    return float(d @ d + np.trace(C1) + np.trace(C2)
                 - 2.0 * np.sqrt(np.clip(ev2, 0, None)).sum())


# --------------------------------------------------------------------------
# simplex-constrained least squares: min_w ||Phi^T w - b||, w >= 0, sum w = 1
# --------------------------------------------------------------------------
def proj_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    r = np.nonzero(cond)[0][-1]
    theta = css[r] / (r + 1.0)
    return np.maximum(v - theta, 0.0)


def simplex_ls(Phi, b, iters=4000, w0=None, verbose=0, tag=""):
    """Accelerated projected gradient for min 0.5||Phi^T w - b||^2 on the simplex.

 Phi: (n, k) candidate feature vectors as rows. Returns (w, residual).
 """
    n, k = Phi.shape
    G_norm = np.linalg.norm(Phi, axis=1).max() ** 2 * 1.0
    # Lipschitz constant of the gradient = ||Phi Phi^T||_2; bound by power iter
    v = np.random.default_rng(0).normal(size=k)
    for _ in range(40):
        v = Phi.T @ (Phi @ v)
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    L = 1.05 * max(float(nv), G_norm, 1e-12)
    step = 1.0 / L
    w = np.full(n, 1.0 / n) if w0 is None else w0.copy()
    y = w.copy()
    t = 1.0
    best_w, best_r = w.copy(), np.inf
    for it in range(iters):
        r = Phi.T @ y - b
        g = Phi @ r
        wn = proj_simplex(y - step * g)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = wn + ((t - 1) / tn) * (wn - w)
        w, t = wn, tn
        if it % 25 == 0 or it == iters - 1:
            res = float(np.linalg.norm(Phi.T @ w - b))
            if res < best_r:
                best_r, best_w = res, w.copy()
            if verbose and it % (verbose) == 0:
                print(f"    [{tag} pgd {it:5d}] resid={res:.3e}", flush=True)
    return best_w, best_r


def fcfw_simplex_ls(Phi, b, max_atoms=769, inner=3000, tol=1e-9, verbose=True):
    """Fully-corrective Frank-Wolfe: sparse witness for Assumption (mixability).

 Adds one atom per outer step and re-solves the simplex LS restricted to the
 active set, so the support size is at most the number of outer steps.
 Caratheodory bounds an EXACT solution by k+1 atoms when b lies in the hull;
 max_atoms defaults to that bound (k = 768).
 """
    n, k = Phi.shape
    S = []
    w = np.zeros(0)
    cur = np.zeros(k)
    res = float(np.linalg.norm(cur - b))
    for t in range(max_atoms):
        r = cur - b
        g = Phi @ r
        j = int(np.argmin(g))
        if j in S:
            # linear-minimisation oracle repeated: no further descent direction
            break
        S.append(j)
        A = np.ascontiguousarray(Phi[S], dtype=np.float64)
        w0 = np.zeros(len(S))
        if t > 0:
            w0[:-1] = w
            w0[-1] = 0.0
            w0 = w0 * 0.98 + 0.02 / len(S)
        w, res = simplex_ls(A, b, iters=inner, w0=w0)
        cur = A.T @ w
        if verbose and (t % 50 == 0 or res < tol):
            print(f"    [fcfw {t:4d} atoms] resid={res:.3e}", flush=True)
        if res < tol:
            break
    idx = np.array(S)
    keep = w > 1e-12
    return idx[keep], w[keep] / w[keep].sum(), res


def simplex_ls_mom2(Phi, Z, b_mu, b_S, iters=2500, lam=1.0, verbose=0, w0=None):
    """Match the mean in R^k AND the second-moment matrix in a d-dim subspace.

 Phi: (n,k) features; Z: (n,d) the same points in whitened top-d real-PCA
 coordinates. Minimises
 ||Phi^T w - b_mu||^2 / s_mu + lam * ||Z^T diag(w) Z - b_S||_F^2 / s_S
 over the simplex by accelerated projected gradient. s_mu, s_S are fixed
 scale factors that put the two blocks in comparable units.
 """
    n, k = Phi.shape
    d = Z.shape[1]
    s_mu = max(float(b_mu @ b_mu), 1e-12)
    s_S = max(float((b_S ** 2).sum()), 1e-12)

    def obj(w):
        rm = Phi.T @ w - b_mu
        rs = (Z * w[:, None]).T @ Z - b_S
        return ((rm @ rm) / s_mu + lam * (rs ** 2).sum() / s_S,
                float(np.linalg.norm(rm)), float(np.linalg.norm(rs)))

    def obj_grad(w):
        rm = Phi.T @ w - b_mu
        rs = (Z * w[:, None]).T @ Z - b_S
        f = (rm @ rm) / s_mu + lam * (rs ** 2).sum() / s_S
        g = 2.0 * (Phi @ rm) / s_mu
        g += lam * 2.0 * ((Z @ rs) * Z).sum(1) / s_S
        return f, g

    w = np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float).copy()
    y = w.copy()
    t = 1.0
    step = None
    best = (np.inf, w.copy(), np.inf, np.inf)
    for it in range(iters):
        f, g = obj_grad(y)
        if step is None:
            step = 1.0 / max(np.abs(g).max() * float(n), 1e-12)
        for _ in range(40):                       # backtracking
            wn = proj_simplex(y - step * g)
            dv = wn - y
            fn = obj(wn)[0]
            if fn <= f + g @ dv + (0.5 / step) * (dv @ dv) + 1e-15:
                break
            step *= 0.5
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = wn + ((t - 1) / tn) * (wn - w)
        w, t = wn, tn
        step *= 1.10
        if it % 25 == 0 or it == iters - 1:
            fw, rmw, rsw = obj(w)
            if fw < best[0]:
                best = (fw, w.copy(), rmw, rsw)
            if verbose and it % verbose == 0:
                print(f"    [mom2 {it:5d}] f={fw:.4e} |dmu|={rmw:.3e} "
                      f"|dS|={rsw:.3e} step={step:.2e}", flush=True)
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------
# audit quantities
# --------------------------------------------------------------------------
def rho_law(frac_exceed, p=TAU_P):
    return frac_exceed / p


def wilson_rho(k, M, p=TAU_P, z=1.96):
    lo, hi = sc.wilson(k, M, z)
    return k / M / p, lo / p, hi / p


def writecsv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# --------------------------------------------------------------------------
# self-test: python analysis_common.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.linalg import sqrtm
    rg = np.random.default_rng(0)
    d = 40
    A = rg.normal(size=(d, d))
    C1 = A @ A.T / d + np.eye(d) * 0.3
    B = rg.normal(size=(d, d))
    C2 = B @ B.T / d + np.eye(d) * 0.5
    m1, m2 = rg.normal(size=d), rg.normal(size=d)
    ours = frechet(m1, C1, m2, C2)
    ref = float(((m1 - m2) @ (m1 - m2)) + np.trace(C1) + np.trace(C2)
                - 2 * np.trace(np.real(sqrtm(C1 @ C2))))
    print(f"[1] frechet vs scipy.sqrtm reference: {ours:.12f} vs {ref:.12f} "
          f"(abs diff {abs(ours-ref):.2e})")
    assert abs(ours - ref) < 1e-8

    print(f"[2] frechet(X, X) = {frechet(m1, C1, m1, C1):.3e} (must be ~0)")
    assert abs(frechet(m1, C1, m1, C1)) < 1e-8

    # identical Gaussians up to a mean shift: FMD must equal ||dmu||^2
    sh = rg.normal(size=d)
    assert abs(frechet(m1, C1, m1 + sh, C1) - sh @ sh) < 1e-8
    print(f"[3] pure mean shift: FMD = ||dmu||^2 = {sh @ sh:.6f}  OK")

    # moments: weighted vs explicit
    X = rg.normal(size=(500, 7))
    w = rg.random(500); w /= w.sum()
    mu, S = moments(X, w=w, chunk=97)
    assert np.allclose(mu, w @ X) and np.allclose(S, (X * w[:, None]).T @ X)
    mu2, S2 = moments(X, chunk=97)
    assert np.allclose(mu2, X.mean(0)) and np.allclose(S2, X.T @ X / len(X))
    print("[4] moments() weighted and unweighted forms OK")

    # mixture moments are linear in the law
    Y = rg.normal(size=(300, 7)) + 2.0
    muY, SY = moments(Y)
    e = 0.137
    mm, SS = mix_moments(mu2, S2, muY, SY, e)
    Z = np.concatenate([X, Y])
    wz = np.r_[np.full(len(X), (1 - e) / len(X)), np.full(len(Y), e / len(Y))]
    mz, Sz = moments(Z, w=wz)
    assert np.allclose(mm, mz) and np.allclose(SS, Sz)
    print("[5] mix_moments() equals the explicit mixture law OK")

    # simplex machinery
    v = rg.normal(size=1000)
    pr = proj_simplex(v)
    assert pr.min() >= -1e-12 and abs(pr.sum() - 1) < 1e-10
    P = rg.normal(size=(4000, 12))
    wt = proj_simplex(rg.random(4000))
    tgt = P.T @ wt
    ww, rr = simplex_ls(P, tgt, iters=3000)
    assert rr < 1e-6, rr
    print(f"[6] simplex_ls recovers an in-hull target: residual {rr:.2e}  OK")
    ii, wf, rf = fcfw_simplex_ls(P, tgt, max_atoms=60, inner=1500, tol=1e-9,
                                verbose=False)
    print(f"[7] fcfw_simplex_ls: residual {rf:.2e} on {len(ii)} atoms "
          f"(target dim 12, Caratheodory bound 13)")
    assert rf < 1e-6, rf
    print("ALL SELF-TESTS PASS")
