"""Audit of one sampler configuration: severities of cached samples, exceedance counts at the
audited depths, and the transport used for matched-severity panels."""
import os
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest

NPY = os.path.expandvars("${DATA_ROOT}/imagenet32_train.npy")
K, NFIT = 50, 200_000
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]
INK, SURFACE = "#33322e", "#fcfcfb"
D = 32 * 32 * 3


def gpd_mle(y):
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return xi, sg


def select_u(r):
    best, best_score = None, -1.0
    for j, q in enumerate(QGRID):
        u = np.quantile(r, q)
        y = r[r > u] - u
        xi, sg = gpd_mle(y)
        ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
        if ksp + 1e-12 * j > best_score:
            best_score, best = ksp + 1e-12 * j, (q, u, y, xi, sg)
    return best


def build_severity(device, t0):
    X8 = np.load(NPY, mmap_mode="r")
    N = X8.shape[0]
    rng = np.random.default_rng(8000)
    perm = rng.permutation(N)
    fit_idx, hold_idx = perm[:NFIT], perm[NFIT:]
    msum = torch.zeros(D, dtype=torch.float64, device=device)
    for a in range(0, len(fit_idx), 50_000):
        ch = torch.tensor(np.ascontiguousarray(
            X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
            dtype=torch.float32, device=device)
        msum += ch.sum(0).double()
    m = (msum / len(fit_idx)).float()
    C = torch.zeros((D, D), dtype=torch.float32, device=device)
    for a in range(0, len(fit_idx), 50_000):
        ch = torch.tensor(np.ascontiguousarray(
            X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
            dtype=torch.float32, device=device) - m
        C += ch.T @ ch
    C /= len(fit_idx)
    lam, V = torch.linalg.eigh(C.double())
    lam, V = lam[-K:].flip(0).float(), V[:, -K:].flip(1).float()
    Wmat = V / torch.sqrt(lam)
    R = np.empty(N, dtype=np.float64)
    Wall = np.empty((N, K), dtype=np.float32)
    for a in range(0, N, 100_000):
        ch = torch.tensor(np.ascontiguousarray(X8[a:a+100_000].reshape(-1, D)),
                          dtype=torch.float32, device=device) - m
        w = ch @ Wmat
        R[a:a+100_000] = torch.linalg.norm(w, dim=1).cpu().numpy()
        Wall[a:a+100_000] = w.cpu().numpy()
    Rf, Rh = R[fit_idx], R[hold_idx]
    uq, u, y, xi, sg = select_u(Rf)
    sev = dict(X8=X8, N=N, m=m, V=V, lam=lam, Wmat=Wmat, R=R, Wall=Wall,
               fit_idx=fit_idx, hold_idx=hold_idx, Rf=Rf, Rh=Rh,
               uq=uq, u=u, xi=xi, sg=sg, zeta=(Rf > u).mean(), rng=rng,
               tau=np.quantile(Rh, 1 - 1e-4), tau3=np.quantile(Rh, 1 - 1e-3))
    print(f"severity layer: q={uq} xi={xi:+.3f} sg={sg:.3f} "
          f"tau(1e-4)={sev['tau']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return sev


def model_severities(Xm_uint8, sev, device):
    M = len(Xm_uint8)
    Wg = np.empty((M, K), dtype=np.float32)
    for a in range(0, M, 50_000):
        ch = torch.tensor(Xm_uint8[a:a+50_000].reshape(-1, D).astype(np.float32),
                          dtype=torch.float32, device=device) - sev["m"]
        Wg[a:a+50_000] = (ch @ sev["Wmat"]).cpu().numpy()
    return np.linalg.norm(Wg, axis=1)


def audit_model(name, Xm_uint8, sev, device, fig_path, nkeep=1000, t0=None):
    t0 = t0 or time.time()
    Rm = model_severities(Xm_uint8, sev, device)
    M = len(Rm)
    out = dict(M=M,
               rate3=(Rm >= sev["tau3"]).mean(), n3=int((Rm >= sev["tau3"]).sum()),
               rate4=(Rm >= sev["tau"]).mean(), n4=int((Rm >= sev["tau"]).sum()))
    print(f"{name} own severity-tail rate ({M} samples): at tau(1e-3) "
          f"{out['rate3']:.2e} (truth 1e-3, n={out['n3']}), at tau(1e-4) "
          f"{out['rate4']:.2e} (truth 1e-4, n={out['n4']})", flush=True)
    print(f"{name} severity quantiles q50/99/999: "
          f"{np.quantile(Rm,[.5,.99,.999]).round(2)}  real fit-split: "
          f"{np.quantile(sev['Rf'],[.5,.99,.999]).round(2)}", flush=True)

    order = np.argsort(Rm)[-nkeep:]
    r_src = Rm[order]
    print(f"kept top {nkeep} of {M} (model-tail severities "
          f"[{r_src.min():.2f}, {r_src.max():.2f}])", flush=True)
    rng, u, xi, sg, tau = sev["rng"], sev["u"], sev["xi"], sev["sg"], sev["tau"]
    Ee = rng.exponential(size=nkeep)
    st = sg + xi * (tau - u)
    r_g = np.sort(tau + st * np.expm1(xi * Ee) / xi)
    Vn = sev["V"].cpu().numpy()
    ln = np.sqrt(sev["lam"].cpu().numpy())
    Xt = Xm_uint8[order].reshape(-1, D).astype(np.float64)
    mt = sev["m"].cpu().numpy().astype(np.float64)
    Wm_np = (sev["V"] / torch.sqrt(sev["lam"])).cpu().numpy().astype(np.float64)
    w = (Xt - mt) @ Wm_np
    oor0 = None
    for it in range(120):
        r_cur = np.linalg.norm(w, axis=1)
        fac = (r_g / r_cur) ** 0.5 if it < 100 else (r_g / r_cur)
        Xt += (((fac - 1.0)[:, None] * w) * ln) @ Vn.T
        if it == 0:
            oor0 = float(((Xt < 0) | (Xt > 255)).mean())
        np.clip(Xt, 0.0, 255.0, out=Xt)
        w = (Xt - mt) @ Wm_np
    r_fin = np.linalg.norm(w, axis=1)
    sev_err = np.abs(r_fin - r_g)
    print(f"transport+projection: raw update leaves box at {oor0:.1%}; final "
          f"|sev-target| med={np.median(sev_err):.4f} "
          f"p95={np.quantile(sev_err,.95):.4f} max={sev_err.max():.3f}; "
          f"within 0.05: {(sev_err < 0.05).mean():.1%}", flush=True)

    R, Rh, Wall, hold_idx = sev["R"], sev["Rh"], sev["Wall"], sev["hold_idx"]
    he3 = hold_idx[Rh >= sev["tau3"]]
    Zh3 = Wall[he3] / R[he3][:, None]
    mB = sev["Rf"] >= np.quantile(sev["Rf"], 0.98)
    Zdir = Wall[sev["fit_idx"]][mB] / sev["Rf"][mB][:, None]
    mZ, sZ = Zdir.mean(0), Zdir.std(0)
    pca2 = np.linalg.svd((Zdir - mZ) / sZ, full_matrices=False)[2][:2]
    Zg = w / np.linalg.norm(w, axis=1, keepdims=True)
    Ph = ((Zh3 - mZ) / sZ) @ pca2.T
    Pg = ((Zg - mZ) / sZ) @ pca2.T
    print(f"shape validation (fit-tail PC basis, n_held={len(he3)}):", flush=True)
    for k_, nm in enumerate(["PC1", "PC2"]):
        print(f"  {nm}: gen mean/sd={Pg[:,k_].mean():+.2f}/{Pg[:,k_].std():.2f}  "
              f"held-out={Ph[:,k_].mean():+.2f}/{Ph[:,k_].std():.2f}", flush=True)
    he = hold_idx[Rh >= tau]
    Rh_e = R[he]
    print(f"severity | >= tau: gen q50/90={np.quantile(r_fin,[.5,.9]).round(2)} "
          f"held-out={np.quantile(Rh_e,[.5,.9]).round(2)}  (n_held={len(he)})",
          flush=True)

    plt.rcParams.update({"font.size": 8.5, "text.color": INK})
    NSHOW = 6
    sev_targets = np.quantile(Rh_e, np.linspace(0.10, 0.95, NSHOW))
    pairs = []
    for s_ in sev_targets:
        ih = int(np.argmin(np.abs(Rh_e - s_)))
        ig = int(np.argmin(np.abs(r_fin - Rh_e[ih])))
        pairs.append((ih, ig))
    fig, axs = plt.subplots(2, NSHOW, figsize=(11.0, 4.1), facecolor=SURFACE)
    for jj, (ih, ig) in enumerate(pairs):
        axs[0, jj].imshow(sev["X8"][he[ih]])
        axs[0, jj].set_title(f"real  R={Rh_e[ih]:.1f}", fontsize=8, pad=2)
        axs[1, jj].imshow(np.clip(Xt[ig], 0, 255).reshape(32, 32, 3).astype(np.uint8))
        axs[1, jj].set_title(f"generated  R={r_fin[ig]:.1f}", fontsize=8, pad=2)
        for row in range(2):
            axs[row, jj].set_xticks([]); axs[row, jj].set_yticks([])
    fig.suptitle(f"{name} shapes carried to GPD-clock severities beyond the "
                 "1-in-10,000 level (real top, transported model samples bottom)",
                 fontsize=10, fontweight="bold")
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {fig_path}   ({time.time()-t0:.0f}s)", flush=True)
    return out
