"""Apply the wrapper to every audited configuration and report raw and wrapped tail ratios."""
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

T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPY = os.path.expandvars("${DATA_ROOT}/imagenet32_train.npy")
DOCS = os.path.expandvars("${PROJECT_ROOT}/docs")
K, NFIT = 50, 200_000
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#33322e", "#8a887f", "#fcfcfb"

X8 = np.load(NPY, mmap_mode="r")
N = X8.shape[0]
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

rng = np.random.default_rng(8000)
perm = rng.permutation(N)
fit_idx, hold_idx = perm[:NFIT], perm[NFIT:]
msum = torch.zeros(D, dtype=torch.float64, device=DEV)
for a in range(0, len(fit_idx), 50_000):
    ch = torch.tensor(np.ascontiguousarray(
        X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
        dtype=torch.float32, device=DEV)
    msum += ch.sum(0).double()
m = (msum / len(fit_idx)).float()
C = torch.zeros((D, D), dtype=torch.float32, device=DEV)
for a in range(0, len(fit_idx), 50_000):
    ch = torch.tensor(np.ascontiguousarray(
        X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
        dtype=torch.float32, device=DEV) - m
    C += ch.T @ ch
C /= len(fit_idx)
lam, V = torch.linalg.eigh(C.double())
lam, V = lam[-K:].flip(0).float(), V[:, -K:].flip(1).float()
Wmat = V / torch.sqrt(lam)
R = np.empty(N, dtype=np.float64)
for a in range(0, N, 100_000):
    ch = torch.tensor(np.ascontiguousarray(X8[a:a+100_000].reshape(-1, D)),
                      dtype=torch.float32, device=DEV) - m
    R[a:a+100_000] = torch.linalg.norm(ch @ Wmat, dim=1).cpu().numpy()
Rf, Rh = R[fit_idx], R[hold_idx]
uq, u, y, xi, sg = select_u(Rf)
zeta = (Rf > u).mean()
Rf_sorted = np.sort(Rf)
print(f"calibration law: q={uq} xi={xi:+.3f} sg={sg:.3f} zeta={zeta:.4f}  "
      f"({time.time()-T0:.0f}s)", flush=True)

def F_data_inv(uv):
    """Quantile of the spliced calibration law (empirical below u, GPD above)."""
    uv = np.asarray(uv, float)
    out = np.empty_like(uv)
    Fu = 1.0 - zeta
    lo = uv <= Fu
    out[lo] = np.quantile(Rf_sorted, np.clip(uv[lo], 0, 1))
    e = 1.0 - uv[~lo]
    out[~lo] = u + (sg / xi) * ((e / zeta) ** (-xi) - 1.0)
    return out

def model_R(cache, key_r="Rm", key_x="X32"):
    cc = np.load(os.path.join(DOCS, cache))
    if key_r in cc.files and cache.startswith("ccvfm"):
        return cc[key_r]
    Xm = cc[key_x].astype(np.float32).reshape(len(cc[key_x]), -1)
    Wg = np.empty((len(Xm), K), dtype=np.float32)
    for a in range(0, len(Xm), 50_000):
        ch = torch.tensor(Xm[a:a+50_000], dtype=torch.float32, device=DEV) - m
        Wg[a:a+50_000] = (ch @ Wmat).cpu().numpy()
    return np.linalg.norm(Wg, axis=1)

MODELS = [("iDDPM (2021)", "iddpm_samples_cache.npz"),
          ("EDM (2022)", "edm_samples_cache.npz"),
          ("EDM2-S (2024)", "edm2_samples_cache.npz"),
          ("EDM2-M (2024)", "edm2m_samples_cache.npz")]
if os.path.exists(os.path.join(DOCS, "sgxl_samples_cache.npz")):
    MODELS.append(("StyleGAN-XL (2022)", "sgxl_samples_cache.npz"))

tau3 = np.quantile(Rh, 1 - 1e-3)
tau4 = np.quantile(Rh, 1 - 1e-4)
tgrid = np.linspace(np.quantile(Rh, 0.5), np.quantile(Rh, 1 - 3e-5), 200)
real_exc = np.array([(Rh >= t).mean() for t in tgrid])

fig, axs = plt.subplots(1, len(MODELS), figsize=(3.6 * len(MODELS), 3.6),
                        facecolor=SURFACE, sharey=True)
plt.rcParams.update({"font.size": 9, "text.color": INK})
if len(MODELS) == 1: axs = [axs]
for ax, (name, cache) in zip(axs, MODELS):
    Rm = model_R(cache)
    M = len(Rm)
    ranks = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    Rw = F_data_inv(ranks)                              # wrapped severities
    for pl, tau_ in [(1e-3, tau3), (1e-4, tau4)]:
        rb = (Rm >= tau_).mean() / pl
        ra = (Rw >= tau_).mean() / pl
        print(f"[{name}] rho({pl:g}): raw={rb:.2f}  wrapped={ra:.2f}  "
              f"(n_raw={(Rm >= tau_).sum()}, n_wrap={(Rw >= tau_).sum()}, M={M})",
              flush=True)
    raw_exc = np.array([(Rm >= t).mean() for t in tgrid])
    wrap_exc = np.array([(Rw >= t).mean() for t in tgrid])
    ax.semilogy(tgrid, real_exc, color=INK, lw=1.8, label="real (held-out)")
    ax.semilogy(tgrid, np.where(raw_exc > 0, raw_exc, np.nan), color=ORANGE,
                lw=1.6, ls="--", label="model, raw")
    ax.semilogy(tgrid, np.where(wrap_exc > 0, wrap_exc, np.nan), color=BLUE,
                lw=1.6, label="model, wrapped")
    ax.axvline(tau3, color=MUTED, lw=0.7, ls=":")
    ax.axvline(tau4, color=MUTED, lw=0.7, ls=":")
    ax.set_title(name, fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel("severity t")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel("P(R ≥ t)")
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Tail recalibration: exceedance curves before and after the wrapper "
             "(dotted lines: the 1-in-1,000 and 1-in-10,000 data levels)",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(DOCS, "recalibration.png"), dpi=150, facecolor=SURFACE,
            bbox_inches="tight")
print(f"wrote recalibration.png   ({time.time()-T0:.0f}s)")
print("DONE", flush=True)
