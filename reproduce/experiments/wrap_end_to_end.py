"""Realize the wrapper on images and compare image-level and remapped tail ratios."""
import os
import sys
import time
import numpy as np
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
R50 = L["R50"]
Rf_sorted = np.sort(R50[fit_idx])
Rh = R50[hold_idx]
u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])
tau3, tau4 = float(L["tau3"]), float(L["tau4"])

b8000 = sc.build_basis(8000, kmax=50, verbose=False)
mt = b8000["m"].numpy().astype(np.float64)
Vn = b8000["V"].numpy().astype(np.float64)
ln = np.sqrt(b8000["lam"].numpy().astype(np.float64))
Wm_np = Vn / ln

rows = []
cachemap = dict(sc.CONFIGS)
for model in ["iDDPM", "EDM"]:
    X = sc.load_cache(cachemap[model])
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[model]}.npz")["W"], 50)
    M = len(Rm)
    v = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    Rw_target = sc.spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)
    region = v > 0.99
    n_reg = int(region.sum())
    # sanity: every raw or target exceedance of tau3 lies inside the region
    assert (Rm[~region] < tau3).all() and (Rw_target[~region] < tau3).all()
    idx = np.where(region)[0]
    Xt = X[idx].reshape(-1, sc.D).astype(np.float64)
    r_g = Rw_target[idx]
    w = (Xt - mt) @ Wm_np
    oor0 = None
    for it in range(120):
        r_cur = np.linalg.norm(w, axis=1)
        fac = (r_g / r_cur) ** 0.5 if it < 100 else (r_g / r_cur)
        Xt += (((fac - 1.0)[:, None] * w) * ln) @ Vn.T
        if it == 0:
            oor0 = float(((Xt < 0) | (Xt > 255)).mean())
        np.clip(Xt, 0.0, 255.0, out=Xt)
        w = (Xt - mt) @ Wm_np
    # realized severities of the WRAPPED SAMPLER = transported region + rest raw
    R_img = Rm.copy()
    R_img[idx] = np.linalg.norm(w, axis=1)
    err = np.abs(R_img[idx] - r_g)
    res = dict(model=model, M=M, n_region=n_reg, oor_first_update=oor0,
               med_abs_err=float(np.median(err)),
               p95_abs_err=float(np.quantile(err, .95)),
               frac_within_005=float((err < 0.05).mean()),
               rho3_remap=float((Rw_target >= tau3).mean() / 1e-3),
               rho3_image=float((R_img >= tau3).mean() / 1e-3),
               rho4_remap=float((Rw_target >= tau4).mean() / 1e-4),
               rho4_image=float((R_img >= tau4).mean() / 1e-4),
               rho3_raw=float((Rm >= tau3).mean() / 1e-3))
    rows.append(res)
    print(f"{model}: region={n_reg} med|err|={res['med_abs_err']:.4f} "
          f"within.05={res['frac_within_005']:.1%} | rho3 raw={res['rho3_raw']:.2f} "
          f"remap={res['rho3_remap']:.2f} image={res['rho3_image']:.2f} | "
          f"rho4 remap={res['rho4_remap']:.2f} image={res['rho4_image']:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del X

with open(f"{OUT}/wrap_end_to_end.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote wrap_end_to_end.csv", flush=True)
