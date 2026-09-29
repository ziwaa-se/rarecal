"""Planted-ratio recovery: mix held-out tail images into a base sampler at a known ratio and audit the mixture."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
rng = np.random.default_rng(42)

L = np.load(f"{sc.P2}/layer_seed8000.npz")
R50 = L["R50"]
fit_idx, hold_idx, N = sc.split_indices(8000)
Rh = R50[hold_idx]
Rbase = sc.radius_k(np.load(f"{sc.P2}/model_w100_stylegan_xl.npz")["W"], 50)

M, B = 100_000, 500
RHOS = [0.3, 1.0, 2.0, 5.0]
DEPTHS = [1e-2, 1e-3]
rows = []

# ---------------- Variant A: fixed the reference split thresholds --------------------------
for p in DEPTHS:
    tau = np.quantile(Rh, 1 - p)
    p_hold = (Rh >= tau).mean()          # exact corpus-relative depth
    for rho_t in RHOS:
        eps = rho_t * p_hold
        rho_true = eps / p_hold          # = rho_t by construction
        k = rng.binomial(M, eps, size=B)
        rho_hat = k / (M * p)
        cover = np.zeros(B, bool)
        widths = np.empty(B)
        for b in range(B):
            lo, hi = sc.wilson(int(k[b]), M)
            cover[b] = lo / p <= rho_true <= hi / p
            widths[b] = (hi - lo) / p
        rows.append(dict(variant="A_fixed_tau", p=p, rho_target=rho_t,
                         rho_true=rho_true, mean_rho_hat=rho_hat.mean(),
                         rel_bias=rho_hat.mean() / rho_true - 1,
                         coverage=cover.mean(), mean_ci_width=widths.mean(),
                         B=B, M=M))
        print(f"A p={p:g} rho={rho_t}: mean={rho_hat.mean():.3f} "
              f"cover={cover.mean():.3f}", flush=True)

# ---------------- Variant B: per-replicate independent truth split -----------
nh = len(hold_idx)
for p in DEPTHS:
    for rho_t in RHOS:
        cover = np.zeros(B, bool)
        rho_hats = np.empty(B)
        rho_trues = np.empty(B)
        for b in range(B):
            pr = rng.permutation(nh)
            h1, h2 = Rh[pr[:nh // 2]], Rh[pr[nh // 2:]]
            tau_audit = np.quantile(h1, 1 - p)     # audit's threshold (H1)
            tau_truth = np.quantile(h2, 1 - p)     # construction/truth (H2)
            p_truth = (h2 >= tau_truth).mean()
            eps = rho_t * p_truth
            pool = h2[h2 >= tau_truth]
            base_tr = Rbase[Rbase < tau_truth]
            # exact mixture exceedance of the audit threshold
            n_pl = rng.binomial(M, eps)
            k_pl = rng.binomial(n_pl, (pool >= tau_audit).mean())
            k_ba = rng.binomial(M - n_pl, (base_tr >= tau_audit).mean())
            k = k_pl + k_ba
            # planted truth wrt H2 law
            q_exc = eps * 1.0 + (1 - eps) * (base_tr >= tau_truth).mean()
            rho_true = q_exc / p_truth
            rho_hats[b] = k / (M * p)
            lo, hi = sc.wilson(int(k), M)
            cover[b] = lo / p <= rho_true <= hi / p
            rho_trues[b] = rho_true
        rows.append(dict(variant="B_split_tau", p=p, rho_target=rho_t,
                         rho_true=rho_trues.mean(), mean_rho_hat=rho_hats.mean(),
                         rel_bias=rho_hats.mean() / rho_trues.mean() - 1,
                         coverage=cover.mean(),
                         mean_ci_width=np.nan, B=B, M=M))
        print(f"B p={p:g} rho={rho_t}: mean={rho_hats.mean():.3f} "
              f"cover={cover.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- image-level end-to-end equivalence check -------------------
p, rho_t = 1e-3, 2.0
tau = np.quantile(Rh, 1 - p)
p_hold = (Rh >= tau).mean()
eps = rho_t * p_hold
rng2 = np.random.default_rng(4242)
n_pl = rng2.binomial(M, eps)
pool_idx = hold_idx[Rh >= tau]                       # real tail image indices
base_ok = np.where(Rbase < tau)[0]                   # truncated base samples
pl_draw = rng2.choice(pool_idx, n_pl, replace=True)
ba_draw = rng2.choice(base_ok, M - n_pl, replace=True)
# severity-level count
k_sev = int((R50[pl_draw] >= tau).sum() + (Rbase[ba_draw] >= tau).sum())
# image-level: assemble mixed images, re-project through the severity map
X8 = np.load(sc.NPY, mmap_mode="r")
Xbase = sc.load_cache(f"{sc.DOCS}/sgxl_samples_cache.npz")
b8000 = sc.build_basis(8000, kmax=50, verbose=False)
Xmix = np.concatenate([np.ascontiguousarray(X8[np.sort(pl_draw)]),
                       Xbase[ba_draw[:20_000]]])    # 20k base subset suffices
Rmix = sc.radius_k(sc.project_images(Xmix, b8000), 50)
k_img_pl = int((Rmix[:n_pl] >= tau).sum())
k_sev_pl = int((R50[np.sort(pl_draw)] >= tau).sum())
k_img_ba = int((Rmix[n_pl:] >= tau).sum())
k_sev_ba = int((Rbase[ba_draw[:20_000]] >= tau).sum())
match = (k_img_pl == k_sev_pl) and (k_img_ba == k_sev_ba)
print(f"image-level check: planted {k_img_pl}=={k_sev_pl}, "
      f"base20k {k_img_ba}=={k_sev_ba}  match={match}", flush=True)
rows.append(dict(variant="image_equivalence", p=p, rho_target=rho_t,
                 rho_true=float(match), mean_rho_hat=k_sev / (M * p),
                 rel_bias=0.0, coverage=float(match), mean_ci_width=np.nan,
                 B=1, M=M))

import csv
with open(f"{OUT}/planted_ratio.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote planted_ratio.csv  ({time.time()-t0:.0f}s)", flush=True)
