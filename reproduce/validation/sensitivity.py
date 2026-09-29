"""Sensitivity of verdicts to the basis dimension K, the split seed and the threshold rule."""
import os
import sys
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")

# ---------- default + K variants (seed 8000) ----------
L0 = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
Wall = np.load(f"{sc.P2}/wall100_seed8000.npy", mmap_mode="r")

rows = []
gpd_rows = []

def audit_variant(variant, Rm_by_config, tau3):
    for name, Rm in Rm_by_config.items():
        a = sc.audit_rho(Rm, tau3, 1e-3)
        rows.append(dict(variant=variant, config=name, M=a["M"], k=a["k"],
                         rho=a["rho"], lo=a["lo"], hi=a["hi"],
                         verdict=sc.verdict(a["lo"], a["hi"])))

W_by_config = {name: np.load(f"{sc.P2}/model_w100_{sc.SLUG[name]}.npz")["W"]
               for name, _ in sc.CONFIGS}

for K, variant in [(50, "default"), (30, "K30"), (100, "K100")]:
    RK = sc.radius_k(np.asarray(Wall), K)
    RfK, RhK = RK[fit_idx], RK[hold_idx]
    tau3 = np.quantile(RhK, 1 - 1e-3)
    q, u, xi, sg, ksp = sc.select_u(RfK)
    zeta = (RfK > u).mean()
    thr4 = sc.spliced_quantile(np.array([1 - 1e-4]), np.sort(RfK), u, xi, sg,
                               zeta)[0]
    e4 = (RhK >= thr4).mean() / 1e-4
    gpd_rows.append(dict(variant=variant, q=q, u=u, xi=xi, sg=sg, ksp=ksp,
                         Finv_1e4=thr4, e_1e4=e4))
    audit_variant(variant, {n: sc.radius_k(W, K) for n, W in W_by_config.items()},
                  tau3)
    print(f"{variant}: q={q} xi={xi:+.3f} ksp={ksp:.2f} e4={e4:.3f}", flush=True)

# ---------- seed variants ----------
for seed in [8001, 8002]:
    Ls = np.load(f"{sc.P2}/layer_seed{seed}.npz")
    tau3 = float(Ls["tau3"])
    Rm_by = {name: np.load(f"{sc.P2}/model_r50_seed{seed}_{sc.SLUG[name]}.npy")
             for name, _ in sc.CONFIGS}
    audit_variant(f"seed{seed}", Rm_by, tau3)
    # GPD axis info for completeness
    fi, hi_, _ = sc.split_indices(seed)
    Rfs = Ls["R50"][fi]; Rhs = Ls["R50"][hi_]
    thr4 = sc.spliced_quantile(np.array([1 - 1e-4]), np.sort(Rfs),
                               float(Ls["u"]), float(Ls["xi"]), float(Ls["sg"]),
                               float(Ls["zeta"]))[0]
    gpd_rows.append(dict(variant=f"seed{seed}", q=float(Ls["q"]), u=float(Ls["u"]),
                         xi=float(Ls["xi"]), sg=float(Ls["sg"]),
                         ksp=float(Ls["ksp"]), Finv_1e4=thr4,
                         e_1e4=(Rhs >= thr4).mean() / 1e-4))
    print(f"seed{seed} done", flush=True)

# ---------- threshold-rule variants (same severities as default) ----------
R50 = sc.radius_k(np.asarray(Wall), 50)
Rf, Rh = R50[fit_idx], R50[hold_idx]
tau3 = np.quantile(Rh, 1 - 1e-3)
for qfix in [0.98, 0.995]:
    variant = f"fixedq{qfix:g}"
    q, u, xi, sg, ksp = sc.fixed_u(Rf, qfix)
    zeta = (Rf > u).mean()
    thr4 = sc.spliced_quantile(np.array([1 - 1e-4]), np.sort(Rf), u, xi, sg,
                               zeta)[0]
    e4 = (Rh >= thr4).mean() / 1e-4
    gpd_rows.append(dict(variant=variant, q=q, u=u, xi=xi, sg=sg, ksp=ksp,
                         Finv_1e4=thr4, e_1e4=e4))
    audit_variant(variant, {n: sc.radius_k(W, 50) for n, W in W_by_config.items()},
                  tau3)
    print(f"{variant}: xi={xi:+.3f} ksp={ksp:.3f} e4={e4:.3f}", flush=True)

with open(f"{OUT}/sensitivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/sensitivity_gpd.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gpd_rows[0].keys()))
    w.writeheader(); w.writerows(gpd_rows)

# ---------- stability criteria ----------
default = {r["config"]: r for r in rows if r["variant"] == "default"}
variants = sorted({r["variant"] for r in rows} - {"default"})
n_total = n_keep = 0
sign_flips = []
adjacency_ok = True
med_drift = {}
lines = []
for cfg in default:
    drifts = []
    for v in variants:
        rv = next(r for r in rows if r["variant"] == v and r["config"] == cfg)
        n_total += 1
        same = rv["verdict"] == default[cfg]["verdict"]
        n_keep += same
        pair = {default[cfg]["verdict"], rv["verdict"]}
        if pair == {"under", "over"}:
            sign_flips.append((cfg, v))
        if not same and "calibrated" not in pair:
            adjacency_ok = False
        drifts.append(abs(np.log(rv["rho"] / default[cfg]["rho"]))
                      if rv["rho"] > 0 and default[cfg]["rho"] > 0 else np.inf)
    med_drift[cfg] = float(np.median(drifts))
    lines.append(f"{cfg:12s} default={default[cfg]['verdict']:10s} "
                 f"rho={default[cfg]['rho']:.3f} median|log drift|="
                 f"{med_drift[cfg]:.4f} (limit {np.log(1.25):.4f}) "
                 f"{'PASS' if med_drift[cfg] <= np.log(1.25) else 'FAIL'}")

s1 = len(sign_flips) == 0
s2 = (n_keep / n_total >= 0.80) and adjacency_ok
s3 = all(v <= np.log(1.25) for v in med_drift.values())
summary = (f" criterion 1, no under/over sign flips: {'PASS' if s1 else 'FAIL ' + str(sign_flips)}\n"
           f"criterion 2, retention {n_keep}/{n_total} = {n_keep/n_total:.1%} "
           f"(>=80%, adjacent-only={adjacency_ok}): {'PASS' if s2 else 'FAIL'}\n"
           f"criterion 3, estimate stability: {'PASS' if s3 else 'FAIL'}\n" + "\n".join(lines))
print("\n" + summary, flush=True)
with open(f"{OUT}/e5_summary.txt", "w") as f:
    f.write(summary + "\n")
print("wrote sensitivity.csv, sensitivity_gpd.csv, e5_summary.txt", flush=True)
