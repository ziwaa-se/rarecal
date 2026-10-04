"""Audit StyleGAN2-ADA and DDPM on CelebA-HQ and apply the wrapper."""
import os
import sys
import csv
import time
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar, brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc  # noqa: E402

T0 = time.time()
P3 = os.path.expandvars("${CACHE_ROOT}/paper")
DATA = os.path.expandvars("${PROJECT_ROOT}/paper/data")
LAYER = f"{P3}/d2_layer_seed8000.npz"
MODELS = [("SG2-ADA", f"{P3}/d2_sg2ada_samples_cache.npz"),
          ("DDPM-DDIM100", f"{P3}/d2_ddpm_samples_cache.npz")]
DEPTHS = [1e-2, 3e-3, 1e-3]
Z = 1.96
CHI90 = chi2.ppf(0.90, 1)


# ---- profile likelihood on tail prob at fixed threshold (gpd_validation) ----
def llA(y, xi, sg):
    if sg <= 0: return -np.inf
    a = 1 + xi * y / sg
    if a.min() <= 0: return -np.inf
    if abs(xi) < 1e-9:
        return -len(y) * np.log(sg) - y.sum() / sg
    return -len(y) * np.log(sg) - (1 + 1/xi) * np.log(a).sum()


def profile_ci(y, zeta, tau_u, p_point):
    def sig_of(xi, p):
        if p >= zeta: return None
        if abs(xi) < 1e-9: return tau_u / np.log(zeta / p)
        dd = (p / zeta) ** (-xi) - 1.0
        return xi * tau_u / dd if dd != 0 else None
    def lstar(p):
        def neg(xi):
            s = sig_of(xi, p)
            return 1e9 if s is None or s <= 0 else -llA(y, xi, s)
        return -minimize_scalar(neg, bounds=(-0.95, 2.0), method="bounded",
                                options=dict(xatol=1e-4)).fun
    p_ref = p_point if p_point > 0 else zeta * 1e-4
    grid = p_ref * 10.0 ** np.linspace(-2, 2, 25)
    ls = np.array([lstar(p) for p in grid])
    k = int(np.argmax(ls)); lmax = ls[k]; lp = np.log10(grid[k])
    def g(logp): return 2 * (lmax - lstar(10 ** logp)) - CHI90
    lo, hi = 0.0, zeta
    for sgn in (-1, 1):
        b_ = lp
        for step in np.arange(0.25, 8, 0.25):
            b_ = lp + sgn * step
            if sgn > 0 and 10 ** b_ >= zeta * 0.999:
                b_ = np.log10(zeta * 0.999); break
            if g(b_) > 0: break
        try:
            root = brentq(g, min(lp, b_), max(lp, b_), xtol=1e-3)
            if sgn < 0: lo = 10 ** root
            else: hi = 10 ** root
        except ValueError:
            pass
    return lo, hi


def quad_ci(rho, lo, hi, p, n_h, z=Z):
    sig_thr = np.sqrt((1 - p) / (n_h * p)) * max(rho, 1e-9)
    add = z * sig_thr
    return (rho - np.sqrt((rho - lo) ** 2 + add ** 2),
            rho + np.sqrt((hi - rho) ** 2 + add ** 2))


# ---------------- layer ----------------
L = np.load(LAYER)
R50, fit_idx, hold_idx = L["R50"], L["fit_idx"], L["hold_idx"]
Rf, Rh = R50[fit_idx], R50[hold_idx]
Rf_sorted = np.sort(Rf)
u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])
taus = {1e-2: float(L["tau2"]), 3e-3: float(L["tau3e3"]), 1e-3: float(L["tau3"])}
tau4 = float(L["tau4"])
n_h = len(hold_idx)
y_exc = Rf[Rf > u] - u  # fit excesses for the profile likelihood

Wmat = torch_W = None


def radii(X32):
    import torch
    m = torch.tensor(L["m"])
    W = torch.tensor(L["V"]) / torch.sqrt(torch.tensor(L["lam"]))
    out = np.empty((len(X32), W.shape[1]), dtype=np.float32)
    for a in range(0, len(X32), 10_000):
        ch = torch.tensor(
            X32[a:a + 10_000].reshape(-1, 3072).astype(np.float32)) - m
        out[a:a + 10_000] = (ch @ W).numpy()
    return np.sqrt((out.astype(np.float64) ** 2).sum(1))


def e_inherited(tau_p, p):
    if tau_p > u:
        return zeta * max(1 + xi * (tau_p - u) / sg, 0.0) ** (-1 / xi) / p
    return float((Rf > tau_p).mean()) / p


# real tail prob at tau4 under the fit GPD: 90% profile interval (shared by
# both models; uncertainty of the extrapolated threshold itself)
p4_lo, p4_hi = profile_ci(y_exc, zeta, tau4 - u, 1e-4)
print(f"profile 90% CI for P_real(R>=tau4={tau4:.3f}): "
      f"[{p4_lo:.2e}, {p4_hi:.2e}] (nominal 1e-4)", flush=True)

lead_rows, wrap_rows = [], []
for name, cache in MODELS:
    if not os.path.exists(cache):
        print(f"{name}: cache missing ({cache}) — skipped", flush=True)
        continue
    X = sc.load_cache(cache)
    if len(X) < int(os.environ.get("MIN_M", "5000")):
        print(f"{name}: cache has only {len(X)} < MIN_M samples — skipped "
              f"(rerun when sampling lands)", flush=True)
        continue
    Rm = radii(X)
    M = len(Rm)
    print(f"\n== {name}  M={M} ==", flush=True)
    for p in DEPTHS:
        a = sc.audit_rho(Rm, taus[p], p)
        qlo, qhi = quad_ci(a["rho"], a["lo"], a["hi"], p, n_h)
        v = sc.verdict(qlo, qhi)
        lead_rows.append(dict(
            model=name, M=M, p=p, threshold=f"{taus[p]:.6f}",
            exceedances=a["k"], rho=f"{a['rho']:.4f}",
            wilson_lo=f"{a['lo']:.4f}", wilson_hi=f"{a['hi']:.4f}",
            quad_lo=f"{qlo:.4f}", quad_hi=f"{qhi:.4f}", verdict=v,
            extrapolation="", profile_lo="", profile_hi=""))
        print(f"  p={p:g}: k={a['k']} rho={a['rho']:.3f} "
              f"wilson=[{a['lo']:.3f},{a['hi']:.3f}] "
              f"quad=[{qlo:.3f},{qhi:.3f}] -> {v}", flush=True)
    # 1e-4 flagged extrapolation
    k4 = int((Rm >= tau4).sum())
    w4lo, w4hi = sc.wilson(k4, M)
    rho4 = k4 / M / 1e-4
    # propagate: model-count Wilson x profile interval on the real tail prob
    plo = (w4lo / p4_hi) if p4_hi > 0 else 0.0
    phi = (w4hi / p4_lo) if p4_lo > 0 else np.inf
    lead_rows.append(dict(
        model=name, M=M, p=1e-4, threshold=f"{tau4:.6f}",
        exceedances=k4, rho=f"{rho4:.4f}",
        wilson_lo=f"{w4lo/1e-4:.4f}", wilson_hi=f"{w4hi/1e-4:.4f}",
        quad_lo="", quad_hi="", verdict="extrapolation_only",
        extrapolation="GPD_FLAGGED", profile_lo=f"{plo:.4f}",
        profile_hi=f"{phi:.4f}"))
    print(f"  p=1e-4 [FLAGGED GPD extrap]: k={k4} rho={rho4:.3f} "
          f"wilson/1e-4=[{w4lo/1e-4:.3f},{w4hi/1e-4:.3f}] "
          f"profile-propagated=[{plo:.3f},{phi:.3f}]", flush=True)
    # WRAP: rank/(M+1) through the calibration law
    v_ranks = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    Rw = sc.spliced_quantile(v_ranks, Rf_sorted, u, xi, sg, zeta)
    for p in DEPTHS:
        wr = float((Rw >= taus[p]).mean()) / p
        e_p = e_inherited(taus[p], p)
        ok_0911 = 0.9 <= wr <= 1.1
        # Prop wrap(i) predicts the wrapped rate EQUALS the calibration layer's
        # own error e(p); the test is |rho_w - e(p)|, NOT whether rho_w is
        # closer to 1 than e(p) is (the earlier formula, which duplicated
        # in_09_11 and read as if it refuted the claim it was checking).
        matches_e = abs(wr - e_p) <= 0.01
        wrap_rows.append(dict(
            model=name, M=M, p=p, wrapped_rho=f"{wr:.4f}",
            self_error_e=f"{e_p:.4f}", dev_vs_self=f"{wr - e_p:+.4f}",
            in_09_11=int(ok_0911), matches_e_within_0p01=int(matches_e)))
        print(f"  WRAP p={p:g}: rho_w={wr:.3f} e(p)={e_p:.3f} "
              f"dev={wr-e_p:+.3f} in[0.9,1.1]={ok_0911} "
              f"matches_e(<=0.01)={matches_e}", flush=True)
    del X, Rm, Rw

if lead_rows:
    with open(f"{DATA}/celebahq_leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lead_rows[0].keys()))
        w.writeheader(); w.writerows(lead_rows)
    with open(f"{DATA}/celebahq_wrapped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrap_rows[0].keys()))
        w.writeheader(); w.writerows(wrap_rows)
    print(f"\nwrote celebahq_leaderboard.csv ({len(lead_rows)} rows), "
          f"celebahq_wrapped.csv ({len(wrap_rows)} rows)  ({time.time()-T0:.0f}s)",
          flush=True)
else:
    print("no caches present yet — nothing written", flush=True)
