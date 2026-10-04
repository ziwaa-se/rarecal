"""Severity functional and calibration law (same numerics as tailaudit/severity.py)."""
import os
import time
import numpy as np
import torch
from scipy.stats import genpareto, kstest

from paths import NPY, LAYER

D = 32 * 32 * 3
NFIT = 200_000
K = 50
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]
Z = 1.96

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "16")))


# ------------------------------------------------------------------ GPD layer
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


def spliced_quantile(uv, Rf_sorted, u, xi, sg, zeta):
    """F_hat^{-1}: fit-split empirical quantile below u (mass 1-zeta), GPD above."""
    uv = np.asarray(uv, float)
    out = np.empty_like(uv)
    Fu = 1.0 - zeta
    lo = uv <= Fu
    out[lo] = np.quantile(Rf_sorted, np.clip(uv[lo], 0, 1))
    e = 1.0 - uv[~lo]
    out[~lo] = u + (sg / xi) * ((e / zeta) ** (-xi) - 1.0)
    return out


# ------------------------------------------------------------------ intervals
def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def audit_rho(Rm, tau, p, n_h, z=Z):
    """rho-hat with Wilson and the threshold-aware quadrature interval."""
    M = len(Rm)
    k = int((Rm >= tau).sum())
    rho = k / M / p
    lw, uw = wilson(k, M, z)
    lw, uw = lw / p, uw / p
    st = np.sqrt((1 - p) / (n_h * p))
    lo = rho - np.sqrt((rho - lw) ** 2 + (z * rho * st) ** 2)
    hi = rho + np.sqrt((uw - rho) ** 2 + (z * rho * st) ** 2)
    return dict(M=M, k=k, rho=rho, wilson_lo=lw, wilson_hi=uw, quad_lo=lo, quad_hi=hi,
                verdict=verdict(lo, hi))


def verdict(lo, hi):
    if hi < 1.0:
        return "under"
    if lo > 1.0:
        return "over"
    return "calibrated"


# ------------------------------------------------------------------ corpus
def split_indices(seed):
    X8 = np.load(NPY, mmap_mode="r")
    N = X8.shape[0]
    perm = np.random.default_rng(seed).permutation(N)
    return perm[:NFIT], perm[NFIT:], N


def build_basis(seed, kmax=K, verbose=True):
    """Fit-split mean + top-kmax whitening basis (torch f32 CPU, reference numerics)."""
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
    return dict(seed=seed, m=m, V=V, lam=lam, fit_idx=fit_idx, hold_idx=hold_idx, N=N)


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
        print(f"[seed {basis['seed']}] reals projected ({time.time()-t0:.0f}s)", flush=True)
    return Wall


def project_images(X_uint8, basis, device=None):
    """Whitened coords (M, kmax) of images (M,32,32,3) or (M,3072) uint8.
 float32 arithmetic exactly as the reference run (GPU or CPU gives the same law)."""
    Xf = X_uint8.reshape(len(X_uint8), -1)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    Wmat = (basis["V"] / torch.sqrt(basis["lam"])).to(dev)
    m = basis["m"].to(dev)
    W = np.empty((len(Xf), Wmat.shape[1]), dtype=np.float32)
    for a in range(0, len(Xf), 50_000):
        ch = torch.tensor(Xf[a:a + 50_000].astype(np.float32), device=dev) - m
        W[a:a + 50_000] = (ch @ Wmat).cpu().numpy()
    return W


def radius_k(W, k=K):
    """K-truncated whitened radius from the coords (float64 accumulate)."""
    return np.sqrt((W[:, :k].astype(np.float64) ** 2).sum(1))


def severities(X_uint8, basis, device=None):
    return radius_k(project_images(X_uint8, basis, device), K)


def load_cache(path):
    X = np.load(path)["X32"]
    return X.reshape(len(X), 32, 32, 3)


# ------------------------------------------------------------------ frozen layer
def load_layer(path=LAYER):
    """Returns (layer dict of scalars/arrays, basis dict of torch tensors)."""
    z = np.load(path)
    L = {k: (float(z[k]) if z[k].shape == () else z[k]) for k in z.files}
    basis = dict(seed=int(L["seed"]), m=torch.tensor(L["m"]), V=torch.tensor(L["V"]),
                 lam=torch.tensor(L["lam"]), fit_idx=L["fit_idx"], hold_idx=L["hold_idx"],
                 N=int(L["N"]))
    return L, basis
