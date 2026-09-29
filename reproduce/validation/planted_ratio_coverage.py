"""Coverage of Wilson-only and quadrature intervals in the independent-threshold planted design."""
import os
import sys
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
rng = np.random.default_rng(123)

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
Rh = L["R50"][hold_idx]
Rbase = sc.radius_k(np.load(f"{sc.P2}/model_w100_stylegan_xl.npz")["W"], 50)

M, B = 100_000, 500
rows = []
for p in [1e-2, 1e-3]:
    for rho_t in [0.3, 1.0, 2.0, 5.0]:
        nh = len(hold_idx)
        n1 = nh // 2
        sig_thr = np.sqrt((1 - p) / (n1 * p))     # delta-method rel. error
        cov_w = np.zeros(B, bool)
        cov_q = np.zeros(B, bool)
        for b in range(B):
            pr = rng.permutation(nh)
            h1, h2 = Rh[pr[:n1]], Rh[pr[n1:]]
            tau_a = np.quantile(h1, 1 - p)
            tau_t = np.quantile(h2, 1 - p)
            p_t = (h2 >= tau_t).mean()
            eps = rho_t * p_t
            pool = h2[h2 >= tau_t]
            base_tr = Rbase[Rbase < tau_t]
            n_pl = rng.binomial(M, eps)
            k = (rng.binomial(n_pl, (pool >= tau_a).mean())
                 + rng.binomial(M - n_pl, (base_tr >= tau_a).mean()))
            q_exc = eps + (1 - eps) * (base_tr >= tau_t).mean()
            rho_true = q_exc / p_t
            rho_hat = k / (M * p)
            lo, hi = sc.wilson(int(k), M)
            lo, hi = lo / p, hi / p
            cov_w[b] = lo <= rho_true <= hi
            # quadrature: widen each half-interval by the threshold term
            add = 1.96 * sig_thr * rho_hat
            lo_q = rho_hat - np.sqrt((rho_hat - lo) ** 2 + add ** 2)
            hi_q = rho_hat + np.sqrt((hi - rho_hat) ** 2 + add ** 2)
            cov_q[b] = lo_q <= rho_true <= hi_q
        rows.append(dict(p=p, rho_target=rho_t, sigma_thr_rel=sig_thr,
                         coverage_wilson=cov_w.mean(),
                         coverage_quadrature=cov_q.mean(), B=B, M=M,
                         n_threshold=n1))
        print(f"p={p:g} rho={rho_t}: wilson={cov_w.mean():.3f} "
              f"quadrature={cov_q.mean():.3f}", flush=True)

with open(f"{OUT}/planted_ratio_coverage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote planted_ratio_coverage.csv", flush=True)
