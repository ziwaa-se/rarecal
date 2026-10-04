"""Cluster-level bootstrap for tail-probability intervals on clustered records (WM-811K)."""
import os
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import genpareto

T0 = time.time()
z = np.load(os.path.expandvars("${PROJECT_ROOT}/code/wafer_features.npz"), allow_pickle=True)
R, lot = z["R"].astype(np.float64), z["lot"]
lots, lot_inv = np.unique(lot, return_inverse=True)
order = np.argsort(lot_inv, kind="stable")
bounds = np.searchsorted(lot_inv[order], np.arange(len(lots)))
idx_by_lot = np.split(order, bounds[1:])
R_by_lot = [R[ix] for ix in idx_by_lot]

def gpd_p(rf, tau):
    u = np.quantile(rf, 0.98)
    y = rf[rf > u] - u
    if len(y) < 30: return np.nan
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return (rf > u).mean() * genpareto.sf(tau - u, xi, 0.0, sg)

P_T = [1e-2, 3e-3, 1e-3]
res = {(nl, p, m): [] for nl in (400, 1600) for p in P_T for m in ("iid", "cl")}
wid = {(nl, p, m): [] for nl in (400, 1600) for p in P_T for m in ("iid", "cl")}
B = 300
for rep in range(20):
    rng = np.random.default_rng(6000 + rep)      # same splits as the main run
    perm = rng.permutation(len(lots))
    for nl in (400, 1600):
        fit_lots = perm[:nl]
        fit_idx = np.concatenate([idx_by_lot[j] for j in fit_lots])
        hold = np.ones(len(R), bool); hold[fit_idx] = False
        Rf, Rh = R[fit_idx], R[hold]
        Rf_lots = [R_by_lot[j] for j in fit_lots]
        taus = {p: np.quantile(Rh, 1 - p) for p in P_T}
        truths = {p: (Rh >= taus[p]).mean() for p in P_T}
        draws_iid = {p: [] for p in P_T}
        draws_cl = {p: [] for p in P_T}
        nf = len(Rf)
        for _ in range(B):
            rb = Rf[rng.integers(0, nf, nf)]
            for p in P_T: draws_iid[p].append(gpd_p(rb, taus[p]))
            pick = rng.integers(0, nl, nl)
            rc = np.concatenate([Rf_lots[j] for j in pick])
            for p in P_T: draws_cl[p].append(gpd_p(rc, taus[p]))
        for p in P_T:
            for m, dr in (("iid", draws_iid[p]), ("cl", draws_cl[p])):
                dd = np.array(dr); dd = dd[np.isfinite(dd)]
                lo, hi = np.quantile(dd, [0.05, 0.95])
                res[(nl, p, m)].append(float(lo <= truths[p] <= hi))
                wid[(nl, p, m)].append(np.log10(hi / lo) if lo > 0 else np.nan)
    print(f"rep {rep} done ({time.time()-T0:.0f}s)", flush=True)

print("\n===== cluster vs iid bootstrap, 90% nominal, 20 lot-splits =====")
for nl in (400, 1600):
    for p in P_T:
        ci = np.mean(res[(nl, p, "iid")]); cc = np.mean(res[(nl, p, "cl")])
        wi = np.nanmedian(wid[(nl, p, "iid")]); wc = np.nanmedian(wid[(nl, p, "cl")])
        print(f"  lots={nl:<5} p={p:<7} cover iid={ci:.2f} cluster={cc:.2f} | "
              f"width(dec) iid={wi:.2f} cluster={wc:.2f}", flush=True)
print("DONE", flush=True)
