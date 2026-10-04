"""Support-preserving transport in the logit chart, illustrated on the WM-811K wafer-map corpus."""
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto, spearmanr, norm as ndist
from scipy.special import logit, expit

T0 = time.time()
NPZ = os.path.expandvars("${PROJECT_ROOT}/code/wafer_features.npz")
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#33322e", "#8a887f", "#fcfcfb"

z = np.load(NPZ, allow_pickle=True)
X, R, lot, ncell = (z["X"].astype(np.float64), z["R"].astype(np.float64),
                    z["lot"], z["n_cell"])
d = X.shape[1]
lots, lot_inv = np.unique(lot, return_inverse=True)
order = np.argsort(lot_inv, kind="stable")
bounds = np.searchsorted(lot_inv[order], np.arange(len(lots)))
idx_by_lot = np.split(order, bounds[1:])
print(f"{len(R):,} wafers, {len(lots):,} lots, d={d}  ({time.time()-T0:.0f}s)", flush=True)

def gpd_mle(y):
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return xi, sg

def dequant(Xr, nc, rg):
    n = np.maximum(nc.astype(np.float64), 1.0)
    k = np.rint(Xr * n)
    return (k + rg.random(Xr.shape)) / (n + 1.0)

def transport_t(PSI, target):
    """Vector t: mean(sigmoid(PSI + t)) = target, per row. Monotone bisection."""
    target = np.broadcast_to(np.asarray(target, float), (PSI.shape[0],))
    lo = np.full(PSI.shape[0], -40.0)
    hi = np.full(PSI.shape[0], 40.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = expit(PSI + mid[:, None]).mean(1)
        gt = m > target
        hi[gt] = mid[gt]; lo[~gt] = mid[~gt]
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- split & severity
rng = np.random.default_rng(6000)
perm = rng.permutation(len(lots))
fit_idx = np.concatenate([idx_by_lot[j] for j in perm[:1600]])
hold = np.ones(len(R), bool); hold[fit_idx] = False

Xf = dequant(X[fit_idx], ncell[fit_idx], rng)
Rf = Xf.mean(1)
u = np.quantile(Rf, 0.98)
xi, sg = gpd_mle(Rf[Rf > u] - u)
zeta = (Rf > u).mean()
uB = np.quantile(Rf, 0.97)
mB = Rf >= uB
rB = Rf[mB]
r0 = np.median(rB)
print(f"severity: xi={xi:+.3f} sg={sg:.4f} u={u:.4f}  n_tail={mB.sum()}  r0={r0:.4f}",
      flush=True)

# ---------------------------------------------------------------- shape in logit chart
Rraw_f = R[fit_idx]
atom_f = Rraw_f[mB] == 1.0                              # raw total-failure wafers
kept = ~atom_f
rS = rB[kept]                                           # shape-fit severities
lotS = lot[fit_idx][mB][kept]
PSI = logit(Xf[mB][kept])
t0 = transport_t(PSI, r0)
Z = PSI + t0[:, None]                                   # states on the r0 level set
chk = np.abs(expit(Z).mean(1) - r0).max()
print(f"shape fit: n={len(rS)} (atoms excluded: {atom_f.sum()})  "
      f"transport check max |sev(Z) - r0| = {chk:.2e}", flush=True)

# nonparametric drift mu_j(r) and common scale c(r): severity-bin estimates,
# linearly interpolated; depth-weighted bin edges so the deep bins stay local
QS = np.array([0.0, 0.30, 0.55, 0.75, 0.88, 0.96, 1.0])
edges = np.quantile(rS, QS)
bix = np.clip(np.searchsorted(edges, rS, side="right") - 1, 0, len(QS) - 2)
centers = np.array([np.median(rS[bix == b]) for b in range(len(QS) - 1)])
binmu = np.vstack([Z[bix == b].mean(0) for b in range(len(QS) - 1)])
print("bin sizes:", np.bincount(bix), " centers:", centers.round(3), flush=True)

def mu_of(r):
    r = np.asarray(r, float)
    return np.column_stack([np.interp(r, centers, binmu[:, j]) for j in range(d)])

E = Z - mu_of(rS)
bin_sd = np.array([E[bix == b].std() for b in range(len(QS) - 1)])
def c_of(r):
    return np.interp(np.asarray(r, float), centers, bin_sd)
Es = E / c_of(rS)[:, None]
print("bin scale c(r):", bin_sd.round(3), flush=True)

# residual law: componentwise PIT -> normal, full correlation (nonparanormal)
n_t = Es.shape[0]
ranks = np.argsort(np.argsort(Es, axis=0), axis=0) + 1.0
G = ndist.ppf(ranks / (n_t + 1.0))
C = np.corrcoef(G.T) + 1e-8 * np.eye(d)
Lc = np.linalg.cholesky(C)

# independence diagnostics: iid spearman (invalid under clustering, shown for
# comparison) and a lot-level permutation test (valid resampling unit)
stat = (G ** 2).sum(1)
ind_p = spearmanr(stat, rS)[1]
ulots, uinv = np.unique(lotS, return_inverse=True)
ls_stat = np.bincount(uinv, stat) / np.bincount(uinv)
ls_r = np.bincount(uinv, rS) / np.bincount(uinv)
obs = abs(spearmanr(ls_stat, ls_r)[0])
rgp = np.random.default_rng(1)
perm_null = np.array([abs(spearmanr(ls_stat, rgp.permutation(ls_r))[0])
                      for _ in range(5000)])
ind_lot_p = (perm_null >= obs).mean()
print(f"indep(|G|^2, r): iid p={ind_p:.2f}  lot-permutation p={ind_lot_p:.2f} "
      f"({len(ulots)} tail lots)", flush=True)

# ---------------------------------------------------------------- generation at 1e-3
Rh_raw = R[hold]
tau = np.quantile(Rh_raw, 1 - 1e-3)
truth_tau = (Rh_raw >= tau).mean()
p_hat_tau = zeta * genpareto.sf(tau - u, xi, 0.0, sg)
M = 20_000
Ee = rng.exponential(size=M)
st = sg + xi * (tau - u)
r_g = tau + st * np.expm1(xi * Ee) / xi

Gg = rng.standard_normal((M, d)) @ Lc.T
Ug = ndist.cdf(Gg)
Eg = np.column_stack([np.quantile(Es[:, j], Ug[:, j]) for j in range(d)])
Zg = mu_of(r_g) + c_of(r_g)[:, None] * Eg               # smoothed drift + scaled resid
tg = transport_t(Zg, r_g)                               # transport to target severity
Xg = expit(Zg + tg[:, None])
oob = float(((Xg < 0) | (Xg > 1)).mean())
sev_err = np.abs(Xg.mean(1) - r_g).max()
print(f"generation: p_hat/truth at 1e-3 = {p_hat_tau/truth_tau:.2f}  "
      f"cells outside [0,1]: {oob:.3%}  max |sev-target| = {sev_err:.2e}  "
      f"min sev gen={Xg.mean(1).min():.4f} (tau={tau:.4f})", flush=True)

# ---------------------------------------------------------------- validation
he_idx = np.where(hold & (R >= tau) & (R < 1.0))[0]     # continuous tail only
Xh = dequant(X[he_idx], ncell[he_idx], rng)
Rh_e = Xh.mean(1)
PSIh = logit(Xh)
Zh = PSIh + transport_t(PSIh, r0)[:, None]
print(f"severity | >= tau (atom excluded): gen q50/90="
      f"{np.quantile(r_g,[.5,.9]).round(3)} held-out="
      f"{np.quantile(Rh_e,[.5,.9]).round(3)}  (n_held={len(he_idx)})", flush=True)

mZ, sZ = Zh.mean(0), Zh.std(0)
pca = np.linalg.svd((Zh - mZ) / sZ, full_matrices=False)[2][:2]
Zg_ref = Zg                                             # generated state at reference
Ph = ((Zh - mZ) / sZ) @ pca.T
Pg = ((Zg_ref - mZ) / sZ) @ pca.T
for k_, nm in enumerate(["PC1", "PC2"]):
    print(f"shape {nm}: gen mean/sd={Pg[:,k_].mean():+.2f}/{Pg[:,k_].std():.2f}  "
          f"held-out={Ph[:,k_].mean():+.2f}/{Ph[:,k_].std():.2f}", flush=True)
# per-cell check in the observable chart: mean cell values
cell_gap = np.abs(Xg.mean(0) - Xh.mean(0))
print(f"per-cell mean |gap|: med={np.median(cell_gap):.3f} max={cell_gap.max():.3f} "
      f"(held-out cell sd med={np.median(Xh.std(0)):.3f})", flush=True)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8.5, "text.color": INK})
NSHOW = 6
sev_targets = np.quantile(Rh_e, np.linspace(0.10, 0.95, NSHOW))
pairs = []
for s in sev_targets:
    ih = int(np.argmin(np.abs(Rh_e - s)))
    ig = int(np.argmin(np.abs(r_g - Rh_e[ih])))
    pairs.append((ih, ig))
print("figure pairs (real, gen):",
      [(round(Rh_e[a], 3), round(r_g[b], 3)) for a, b in pairs], flush=True)

n_rad, n_ang = int(z["n_rad"]), int(z["n_ang"])
theta_f = np.linspace(0, 2 * np.pi, n_ang * 80 + 1)
radii = np.sqrt(np.linspace(0, 1, n_rad + 1))
TH, RD = np.meshgrid(theta_f, radii)
def cellgrid(vec):
    g = vec.reshape(n_rad, n_ang)
    return np.repeat(g, 80, axis=1)
fig, axs = plt.subplots(2, NSHOW, figsize=(11.5, 4.4), facecolor=SURFACE,
                        subplot_kw=dict(projection="polar"))
for jj, (ih, ig) in enumerate(pairs):
    for row, (vec, sev, ttl) in enumerate([
            (Xh[ih], Rh_e[ih], "real"), (Xg[ig], r_g[ig], "generated")]):
        ax = axs[row, jj]
        ax.pcolormesh(TH, RD, cellgrid(vec), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ttl}  R={sev:.3f}", fontsize=8, pad=2)
        ax.grid(False)
fig.suptitle("Yield excursions beyond the 1-in-1000 level: held-out real wafers (top) "
             "and generated wafers (bottom) at matched severities",
             fontsize=10, fontweight="bold")
fig.savefig(os.path.expandvars("${PROJECT_ROOT}/docs/wafer_support_chart.png"),
            dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote wafer_support_chart.png   ({time.time()-T0:.0f}s)")
print("DONE", flush=True)
