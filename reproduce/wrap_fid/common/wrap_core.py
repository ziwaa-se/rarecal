"""The wrapper on images: rank remap, damped alternating projection and 8-bit quantization."""
import numpy as np
import torch

from severity import spliced_quantile, D

ITERS, ITERS_DAMPED = 120, 100


def rank_remap(Rm, Rf_sorted, u, xi, sg, zeta):
    M = len(Rm)
    v = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    return v, spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)


def transport(X_uint8, r_target, basis, device, mode="damped", chunk=10_000):
    """Returns (Xt float64 (n,3072) in [0,255], r_final (n,), oor0).

 oor0 = fraction of pixel coordinates the FIRST raw update pushes outside
 the box, before any projection (the paper's oor_0 statistic).
 """
    dev = torch.device(device)
    m = basis["m"].to(dev, torch.float64)
    V = basis["V"].to(dev, torch.float64)                    # (D, K)
    ln = torch.sqrt(basis["lam"].to(dev, torch.float64))     # (K,)
    Wm = V / ln                                              # (D, K) whitening
    VT = V.T.contiguous()                                    # (K, D)
    n = len(X_uint8)
    Xf = X_uint8.reshape(n, D)
    out = np.empty((n, D), dtype=np.float64)
    r_fin = np.empty(n, dtype=np.float64)
    oor_cnt = 0
    for a in range(0, n, chunk):
        Xt = torch.tensor(Xf[a:a + chunk], dtype=torch.float64, device=dev)
        r_g = torch.tensor(r_target[a:a + chunk], dtype=torch.float64, device=dev)
        w = (Xt - m) @ Wm
        n_it = ITERS if mode == "damped" else 1
        for it in range(n_it):
            r_cur = torch.linalg.norm(w, dim=1)
            if mode == "damped" and it < ITERS_DAMPED:
                fac = torch.sqrt(r_g / r_cur)
            else:
                fac = r_g / r_cur
            Xt += (((fac - 1.0)[:, None] * w) * ln) @ VT
            if it == 0:
                oor_cnt += int(((Xt < 0) | (Xt > 255)).sum().item())
            Xt.clamp_(0.0, 255.0)
            w = (Xt - m) @ Wm
        out[a:a + chunk] = Xt.cpu().numpy()
        r_fin[a:a + chunk] = torch.linalg.norm(w, dim=1).cpu().numpy()
    return out, r_fin, oor_cnt / (n * D)


def quantize(Xt_float):
    """Float image -> uint8 (the deployable object; the paper left it float)."""
    return np.clip(np.rint(Xt_float), 0, 255).astype(np.uint8)


def build_arm(arm, X, Rm, Rw, v, basis, device):
    """Produce the uint8 sample set of one arm plus transport diagnostics.

 arm in {raw, wrap-tail, wrap-full, wrap-tail-hardclip}. X (M,32,32,3) uint8.
 Returns (X_arm uint8 (M,32,32,3), info dict).
 """
    M = len(X)
    if arm == "raw":
        return X, dict(n_transported=0)
    if arm in ("wrap-tail", "wrap-tail-hardclip"):
        region = v > 0.99
        # every raw or target exceedance of tau3 must lie inside the region
        idx = np.where(region)[0]
    elif arm == "wrap-full":
        idx = np.arange(M)
    else:
        raise ValueError(arm)
    mode = "hardclip" if arm.endswith("hardclip") else "damped"
    Xt, r_fin, oor0 = transport(X[idx], Rw[idx], basis, device, mode=mode)
    err = np.abs(r_fin - Rw[idx])
    Xa = X.copy()
    Xa[idx] = quantize(Xt).reshape(-1, 32, 32, 3)
    info = dict(n_transported=int(len(idx)), oor0=float(oor0),
                sev_err_med=float(np.median(err)), sev_err_p95=float(np.quantile(err, .95)),
                sev_err_max=float(err.max()), sev_within_005=float((err < 0.05).mean()),
                transported_idx=idx, r_target=Rw[idx])
    return Xa, info
