"""Self-validation: audit the calibration split in the sampler's role over five split seeds, and compute the calibration-law ratio e(p)."""
import os
import sys
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/paper/data")
DEPTHS = [1e-2, 3e-3, 1e-3]

rows = []
err_rows = []
for seed in [8000, 8001, 8002, 8003, 8004]:
    L = np.load(f"{sc.P2}/layer_seed{seed}.npz")
    R = L["R50"]
    fit_idx, hold_idx, _ = sc.split_indices(seed)
    Rf, Rh = R[fit_idx], R[hold_idx]
    # --- 1. explicit self-audit: fit split in the model role ---
    for p in DEPTHS:
        tau = np.quantile(Rh, 1 - p)
        a = sc.audit_rho(Rf, tau, p)
        rows.append(dict(seed=seed, p=p, tau=tau, k=a["k"], M=a["M"],
                         rho_self=a["rho"], lo=a["lo"], hi=a["hi"],
                         covers_1=int(a["lo"] <= 1.0 <= a["hi"])))
        print(f"seed {seed} p={p:g}: rho_self={a['rho']:.3f} "
              f"[{a['lo']:.3f},{a['hi']:.3f}] k={a['k']} "
              f"covers1={a['lo'] <= 1 <= a['hi']}", flush=True)
    # --- 2. calibration-law self-error, both computation paths ---
    Rf_sorted = np.sort(Rf)
    u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])
    for p in [1e-3, 1e-4]:
        # path 1 (forward formula): threshold = F_hat^{-1}(1-p), hold exceedance
        thr = sc.spliced_quantile(np.array([1 - p]), Rf_sorted, u, xi, sg, zeta)[0]
        e_fwd = (Rh >= thr).mean() / p
        # path 2 (wrapped-rho, M=100k rank grid as in recalibration)
        M = 100_000
        v = np.arange(1, M + 1) / (M + 1.0)
        Rw = sc.spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)
        tau_p = np.quantile(Rh, 1 - p)
        e_wrap = (Rw >= tau_p).mean() / p
        # path 3 (continuous inherited error): (1 - F_hat(tau_p)) / p — the
        # exceedance rate of the wrapped LAW itself at the hold threshold
        if tau_p > u:
            e_inh = zeta * (1 + xi * (tau_p - u) / sg) ** (-1 / xi) / p
        else:
            e_inh = (Rf > tau_p).mean() / p
        err_rows.append(dict(seed=seed, p=p, Fhat_inv_thr=thr,
                             e_forward=e_fwd, e_wrapped_rankgrid=e_wrap,
                             e_inherited_continuous=e_inh))
        print(f"seed {seed} e({p:g}): forward formula={e_fwd:.3f} "
              f"wrapped-rankgrid={e_wrap:.3f} inherited-cont={e_inh:.3f}",
              flush=True)

with open(f"{OUT}/self_validation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/self_error.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(err_rows[0].keys()))
    w.writeheader(); w.writerows(err_rows)

cells = sum(r["covers_1"] for r in rows)
print(f"\nself-validation coverage: {cells}/15 seed x depth cells cover 1 "
      f"(target: >= 14)", flush=True)

# --- 3. wrapped model-independence (the reference split law) ---
L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
Rf_sorted = np.sort(L["R50"][fit_idx])
Rh = L["R50"][hold_idx]
u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])
tau3, tau4 = float(L["tau3"]), float(L["tau4"])

wrows = []
wrapped_by_M = {}
for name, _ in sc.CONFIGS:
    suf = "_final" if name in ("DiT-cfg1.5", "DiT-cfg1.0") else ""
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[name]}{suf}.npz")["W"], 50)
    M = len(Rm)
    v = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    Rw = sc.spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)
    key = M
    Rw_sorted = np.sort(Rw)
    if key in wrapped_by_M:
        maxdiff = np.abs(Rw_sorted - wrapped_by_M[key]).max()
    else:
        wrapped_by_M[key] = Rw_sorted
        maxdiff = 0.0
    wrows.append(dict(model=name, M=M,
                      rho3_raw=(Rm >= tau3).mean() / 1e-3,
                      rho3_wrapped=(Rw >= tau3).mean() / 1e-3,
                      rho4_raw=(Rm >= tau4).mean() / 1e-4,
                      rho4_wrapped=(Rw >= tau4).mean() / 1e-4,
                      max_diff_vs_same_M=maxdiff))
    print(f"{name:12s} M={M:6d} wrapped rho3={wrows[-1]['rho3_wrapped']:.3f} "
          f"rho4={wrows[-1]['rho4_wrapped']:.3f} "
          f"maxdiff_same_M={maxdiff:.2e}", flush=True)

with open(f"{OUT}/wrapped_independence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
    w.writeheader(); w.writerows(wrows)
print("wrote self_validation.csv, self_error.csv, wrapped_independence.csv",
      flush=True)
