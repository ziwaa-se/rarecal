"""Component ablations: generalized-Pareto splice, transport chart and projection."""
import os
import sys
import time
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
rows = []

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
R50 = L["R50"]
Rf, Rh = R50[fit_idx], R50[hold_idx]
Rf_sorted = np.sort(Rf)
u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])

# ---------------- (a) GPD on/off: audit-side deep thresholds ----------------
for p in [1e-4, 1e-5]:
    tau_true = np.quantile(Rh, 1 - p)
    thr_gpd = sc.spliced_quantile(np.array([1 - p]), Rf_sorted, u, xi, sg,
                                  zeta)[0]
    thr_emp = np.quantile(Rf, 1 - p)          # raw fit-split quantile [OFF]
    for lab, thr in [("gpd_on", thr_gpd), ("gpd_off_empirical", thr_emp)]:
        # rho-error a calibrated sampler would show at the estimated threshold
        ratio = (Rh >= thr).mean() / p
        rows.append(dict(component="gpd_audit_threshold", condition=f"p={p:g}",
                         variant=lab, value1=thr, value2=tau_true,
                         metric=f"induced rho-error at estimated threshold",
                         result=ratio))
        print(f"p={p:g} {lab}: thr={thr:.3f} (true {tau_true:.3f}) "
              f"induced-rho={ratio:.3f}", flush=True)

# ---------------- (a) GPD on/off: wrapper side ------------------------------
def wrapped_rho(Rm, p, use_gpd):
    M = len(Rm)
    v = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    if use_gpd:
        Rw = sc.spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)
    else:
        Rw = np.quantile(Rf_sorted, v)        # empirical-only calibration law
    tau_p = np.quantile(Rh, 1 - p)
    return (Rw >= tau_p).mean() / p

for model, cond in [("StyleGAN-XL", "clean_calibrated"),
                    ("iDDPM", "adversarial_truncated"),
                    ("EDM", "adversarial_overshoot")]:
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[model]}.npz")["W"], 50)
    for p in [1e-3, 1e-4]:
        for lab, ug in [("gpd_on", True), ("gpd_off_empirical", False)]:
            wr = wrapped_rho(Rm, p, ug)
            rows.append(dict(component="gpd_wrapper", condition=f"{cond},p={p:g}",
                             variant=lab, value1=np.nan, value2=np.nan,
                             metric="wrapped rho", result=wr))
            print(f"{model} p={p:g} {lab}: wrapped rho={wr:.3f}", flush=True)

# ---------------- (b) transport chart on/off --------------------------------
b8000 = sc.build_basis(8000, kmax=50, verbose=False)
mt = b8000["m"].numpy().astype(np.float64)
Vn = b8000["V"].numpy().astype(np.float64)
ln = np.sqrt(b8000["lam"].numpy().astype(np.float64))
Wm_np = Vn / ln
tau4 = float(L["tau4"])

def transport(Xt0, r_g, mode):
    """Return (final |sev-target| stats, oor at first update)."""
    Xt = Xt0.copy()
    w = (Xt - mt) @ Wm_np
    oor0 = None
    if mode == "full":            # reported: damped alternating projection
        iters = 120
        for it in range(iters):
            r_cur = np.linalg.norm(w, axis=1)
            fac = (r_g / r_cur) ** 0.5 if it < 100 else (r_g / r_cur)
            Xt += (((fac - 1.0)[:, None] * w) * ln) @ Vn.T
            if it == 0:
                oor0 = float(((Xt < 0) | (Xt > 255)).mean())
            np.clip(Xt, 0.0, 255.0, out=Xt)
            w = (Xt - mt) @ Wm_np
    elif mode == "oneshot":       # chart scaling once + single clip
        r_cur = np.linalg.norm(w, axis=1)
        fac = r_g / r_cur
        Xt += (((fac - 1.0)[:, None] * w) * ln) @ Vn.T
        oor0 = float(((Xt < 0) | (Xt > 255)).mean())
        np.clip(Xt, 0.0, 255.0, out=Xt)
        w = (Xt - mt) @ Wm_np
    elif mode == "pixel":         # no chart: naive pixel scaling + clip
        r_cur = np.linalg.norm(w, axis=1)
        alpha = (r_g / r_cur)[:, None]
        Xt = mt + alpha * (Xt - mt)
        oor0 = float(((Xt < 0) | (Xt > 255)).mean())
        np.clip(Xt, 0.0, 255.0, out=Xt)
        w = (Xt - mt) @ Wm_np
    r_fin = np.linalg.norm(w, axis=1)
    err = np.abs(r_fin - r_g)
    return dict(med=float(np.median(err)), p95=float(np.quantile(err, .95)),
                within=float((err < 0.05).mean()), oor0=oor0)

rng = np.random.default_rng(8000)  # match audit's post-build rng state? no — fresh, disclosed
for model in ["EDM", "iDDPM"]:
    cachemap = dict(sc.CONFIGS)
    X = sc.load_cache(cachemap[model])
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[model]}.npz")["W"], 50)
    nkeep = 1000
    order = np.argsort(Rm)[-nkeep:]
    Ee = rng.exponential(size=nkeep)
    st = sg + xi * (tau4 - u)
    r_g = np.sort(tau4 + st * np.expm1(xi * Ee) / xi)
    Xt0 = X[order].reshape(-1, sc.D).astype(np.float64)
    for mode in ["full", "oneshot", "pixel"]:
        s = transport(Xt0, r_g, mode)
        rows.append(dict(component="transport_chart", condition=model,
                         variant=mode, value1=s["med"], value2=s["p95"],
                         metric="med|sev-target| / p95 ; result=frac within .05",
                         result=s["within"]))
        print(f"{model} transport[{mode}]: med={s['med']:.4f} p95={s['p95']:.3f} "
              f"within.05={s['within']:.1%} oor0={s['oor0']:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)
    del X

with open(f"{OUT}/ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote ablation.csv", flush=True)
