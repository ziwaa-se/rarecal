"""Severity functionals and calibration law: whitened PCA radius, adaptive generalized-Pareto
threshold, spliced quantile, Wilson interval, tail-calibration ratio and verdict rule."""
import os
import time
import numpy as np
import torch
from scipy.stats import genpareto, kstest

NPY = os.path.expandvars("${DATA_ROOT}/imagenet32_train.npy")
DOCS = os.path.expandvars("${PROJECT_ROOT}/docs")
TFP = os.path.expandvars("${CACHE_ROOT}")
P2 = os.path.expandvars("${CACHE_ROOT}/validation")
D = 32 * 32 * 3
NFIT = 200_000
KMAX = 100
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]

torch.set_num_threads(int(os.environ.get("SEV_THREADS", "16")))

# (name, cache path, final?) -- edmchurn is final at its paused M; DiT from partial
CONFIGS = [
    ("iDDPM",      f"{TFP}/iddpm_samples_cache.npz"),
    ("EDM",        f"{TFP}/edm_samples_cache.npz"),
    ("EDM-churn",  f"{DOCS}/edmchurn_samples_cache.npz"),
    ("EDM2-S",     f"{TFP}/edm2_samples_cache.npz"),
    ("EDM2-M",     f"{TFP}/edm2m_samples_cache.npz"),
    ("StyleGAN-XL",f"{DOCS}/sgxl_samples_cache.npz"),
    ("DiT-cfg1.5", f"{P2}/dit_cfg1p5_partial.npz"),
    ("DiT-cfg1.0", f"{P2}/dit_cfg1_partial.npz"),
]
SLUG = {n: n.lower().replace("-", "_").replace(".", "") for n, _ in CONFIGS}


def gpd_mle(y):
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return xi, sg


def select_u(r, qgrid=QGRID):
    """Adaptive threshold: max KS p-value over qgrid, ties toward deeper q."""
    best, best_score = None, -1.0
    for j, q in enumerate(qgrid):
        u = np.quantile(r, q)
        y = r[r > u] - u
        xi, sg = gpd_mle(y)
        ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
        if ksp + 1e-12 * j > best_score:
            best_score, best = ksp + 1e-12 * j, (q, u, xi, sg, ksp)
    return best


def fixed_u(r, q):
    u = np.quantile(r, q)
    y = r[r > u] - u
    xi, sg = gpd_mle(y)
    ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
    return q, u, xi, sg, ksp


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def split_indices(seed):
    X8 = np.load(NPY, mmap_mode="r")
    N = X8.shape[0]
    perm = np.random.default_rng(seed).permutation(N)
    return perm[:NFIT], perm[NFIT:], N


def build_basis(seed, kmax=KMAX, verbose=True):
    """Fit-split mean + top-kmax whitening basis, torch f32 CPU, audit numerics."""
    t0 = time.time()
    X8 = np.load(NPY, mmap_mode="r")
    fit_idx, hold_idx, N = split_indices(seed)
    dev = torch.device("cpu")
    msum = torch.zeros(D, dtype=torch.float64, device=dev)
    for a in range(0, len(fit_idx), 50_000):
        ch = torch.tensor(np.ascontiguousarray(
            X8[np.sort(fit_idx[a:a + 50_000])].reshape(-1, D)),
            dtype=torch.float32, device=dev)
        msum += ch.sum(0).double()
    m = (msum / len(fit_idx)).float()
    C = torch.zeros((D, D), dtype=torch.float32, device=dev)
    for a in range(0, len(fit_idx), 50_000):
        ch = torch.tensor(np.ascontiguousarray(
            X8[np.sort(fit_idx[a:a + 50_000])].reshape(-1, D)),
            dtype=torch.float32, device=dev) - m
        C += ch.T @ ch
    C /= len(fit_idx)
    lam, V = torch.linalg.eigh(C.double())
    lam = lam[-kmax:].flip(0).float()
    V = V[:, -kmax:].flip(1).float()
    if verbose:
        print(f"[seed {seed}] basis built ({time.time()-t0:.0f}s)", flush=True)
    return dict(seed=seed, m=m, V=V, lam=lam, fit_idx=fit_idx,
                hold_idx=hold_idx, N=N)


def project_reals(basis, verbose=True):
    """Whitened coords (N, kmax) of all reals under the basis."""
    t0 = time.time()
    X8 = np.load(NPY, mmap_mode="r")
    N = basis["N"]
    Wmat = basis["V"] / torch.sqrt(basis["lam"])
    Wall = np.empty((N, Wmat.shape[1]), dtype=np.float32)
    for a in range(0, N, 100_000):
        ch = torch.tensor(np.ascontiguousarray(X8[a:a + 100_000].reshape(-1, D)),
                          dtype=torch.float32) - basis["m"]
        Wall[a:a + 100_000] = (ch @ Wmat).numpy()
    if verbose:
        print(f"[seed {basis['seed']}] reals projected ({time.time()-t0:.0f}s)",
              flush=True)
    return Wall


def project_images(X_uint8, basis):
    """Whitened coords (M, kmax) of model images (M,32,32,3) or (M,3072) uint8."""
    Xf = X_uint8.reshape(len(X_uint8), -1)
    Wmat = basis["V"] / torch.sqrt(basis["lam"])
    W = np.empty((len(Xf), Wmat.shape[1]), dtype=np.float32)
    for a in range(0, len(Xf), 50_000):
        ch = torch.tensor(Xf[a:a + 50_000].astype(np.float32)) - basis["m"]
        W[a:a + 50_000] = (ch @ Wmat).numpy()
    return W


def radius_k(W, k):
    """K-truncated whitened radius from kmax coords (float64 accumulate)."""
    return np.sqrt((W[:, :k].astype(np.float64) ** 2).sum(1))


def load_cache(path):
    cc = np.load(path)
    X = cc["X32"]
    return X.reshape(len(X), 32, 32, 3)


def spliced_quantile(uv, Rf_sorted, u, xi, sg, zeta):
    """F_hat^{-1}: empirical below u (mass 1-zeta), GPD splice above. Mirrors
 recalibration.F_data_inv."""
    uv = np.asarray(uv, float)
    out = np.empty_like(uv)
    Fu = 1.0 - zeta
    lo = uv <= Fu
    out[lo] = np.quantile(Rf_sorted, np.clip(uv[lo], 0, 1))
    e = 1.0 - uv[~lo]
    out[~lo] = u + (sg / xi) * ((e / zeta) ** (-xi) - 1.0)
    return out


def audit_rho(Rm, tau, p, z=1.96):
    """Point estimate + Wilson CI for rho(p) given severity sample and threshold."""
    M = len(Rm)
    k = int((Rm >= tau).sum())
    lo, hi = wilson(k, M, z)
    return dict(M=M, k=k, rho=k / M / p, lo=lo / p, hi=hi / p)


def verdict(lo, hi):
    if hi < 1.0:
        return "under"
    if lo > 1.0:
        return "over"
    return "calibrated"
