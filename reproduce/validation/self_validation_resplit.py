"""Coverage of the interval when the functional and calibration law are fixed and only the split is redrawn."""
import os
import sys
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
rng = np.random.default_rng(7)

R = np.load(f"{sc.P2}/layer_seed8000.npz")["R50"]
N = len(R)
NF = 200_000
B = 400
rows = []
for p in [1e-2, 3e-3, 1e-3]:
    z = 1.96
    rhos = np.empty(B)
    cov_w = np.zeros(B, bool)
    cov_q = np.zeros(B, bool)
    wil_sd = np.empty(B)
    for b in range(B):
        perm = rng.permutation(N)
        Rf, Rh = R[perm[:NF]], R[perm[NF:]]
        tau = np.quantile(Rh, 1 - p)
        k = int((Rf >= tau).sum())
        rho = k / (NF * p)
        lo, hi = sc.wilson(k, NF)
        lo, hi = lo / p, hi / p
        cov_w[b] = lo <= 1.0 <= hi
        # quadrature: add threshold/finite-split term
        sig_thr = np.sqrt((1 - p) / ((N - NF) * p)) * max(rho, 1e-9)
        add = z * sig_thr
        lo_q = rho - np.sqrt((rho - lo) ** 2 + add ** 2)
        hi_q = rho + np.sqrt((hi - rho) ** 2 + add ** 2)
        cov_q[b] = lo_q <= 1.0 <= hi_q
        rhos[b] = rho
        wil_sd[b] = (hi - lo) / (2 * z)
    design = rhos.std(ddof=1) / wil_sd.mean()
    rows.append(dict(p=p, B=B, mean_rho_self=rhos.mean(),
                     sd_rho_self=rhos.std(ddof=1),
                     wilson_implied_sd=wil_sd.mean(),
                     design_effect=design,
                     coverage_wilson=cov_w.mean(),
                     coverage_quadrature=cov_q.mean()))
    print(f"p={p:g}: mean={rhos.mean():.4f} sd={rhos.std(ddof=1):.4f} "
          f"(wilson-implied {wil_sd.mean():.4f}, design x{design:.2f}) "
          f"cover_w={cov_w.mean():.3f} cover_q={cov_q.mean():.3f}", flush=True)

with open(f"{OUT}/self_validation_resplit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote self_validation_resplit.csv", flush=True)
