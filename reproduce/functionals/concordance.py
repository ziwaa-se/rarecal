"""Compare verdicts across R1, R2 and R3: agreement counts and Kendall rank correlations."""
import os
import sys
import numpy as np
import csv
from scipy.stats import kendalltau
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
tau1 = float(L["tau3"])
R2 = np.load(f"{sc.P2}/r2_reals.npy")
R3 = np.load(f"{sc.P2}/r3_reals.npy")
tau2 = np.quantile(R2[hold_idx], 1 - 1e-3)
tau3_ = np.quantile(R3[hold_idx], 1 - 1e-3)

rows = []
rhos = {"S1": [], "S2": [], "S3": []}
verd = {"S1": [], "S2": [], "S3": []}
for name, _ in sc.CONFIGS:
    r1 = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[name]}.npz")["W"], 50)
    r2 = np.load(f"{sc.P2}/r2_{sc.SLUG[name]}.npy")
    r3 = np.load(f"{sc.P2}/r3_{sc.SLUG[name]}.npy")
    row = dict(config=name, M=len(r1))
    for lab, r, t in [("S1", r1, tau1), ("S2", r2, tau2), ("S3", r3, tau3_)]:
        a = sc.audit_rho(r, t, 1e-3)
        v = sc.verdict(a["lo"], a["hi"])
        row[f"rho_{lab}"] = a["rho"]
        row[f"lo_{lab}"] = a["lo"]
        row[f"hi_{lab}"] = a["hi"]
        row[f"verdict_{lab}"] = v
        rhos[lab].append(a["rho"])
        verd[lab].append(v)
    rows.append(row)
    print(f"{name:12s} S1={row['rho_S1']:.2f}({row['verdict_S1'][0]}) "
          f"S2={row['rho_S2']:.2f}({row['verdict_S2'][0]}) "
          f"S3={row['rho_S3']:.2f}({row['verdict_S3'][0]})", flush=True)

pairs = [("S1", "S2"), ("S1", "S3"), ("S2", "S3")]
summary = []
for a, b in pairs:
    agree = sum(x == y for x, y in zip(verd[a], verd[b]))
    tb = kendalltau(rhos[a], rhos[b]).statistic
    summary.append(f"{a}-{b}: verdict agreement {agree}/8, "
                   f"Kendall tau_b={tb:+.2f}")
    print(summary[-1], flush=True)

with open(f"{OUT}/functional_concordance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/functional_concordance_summary.txt", "w") as f:
    f.write("\n".join(summary) + "\n")
print("wrote functional_concordance.csv", flush=True)
