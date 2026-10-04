"""Wrap each configuration's samples and measure FID, KID, FMD and image-level tail ratios before and after."""
import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import NPY, CONFIGS, SLUG, M_OF, FEATS, RESULTS, WRAPPED, cache_path   # noqa: E402
import severity as sc                                                              # noqa: E402
from wrap_core import rank_remap, build_arm                                        # noqa: E402
import feats as ft                                                                 # noqa: E402

ARMS = ["raw", "wrap-tail", "wrap-full", "wrap-tail-hardclip"]
REAL_SPLIT_SEED = 20260816        # REF/POOL cut of the hold split
SUB_SEED = 20260915               # fixed FID subsample per config
POOL_SEED = 20260914              # the REAL-POOL pseudo-sampler

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="raw,wrap-tail")
ap.add_argument("--configs", default="all,real")
ap.add_argument("--nsub", type=int, default=50_000)
ap.add_argument("--m-real", type=int, default=100_000)
ap.add_argument("--tag", default="main")
ap.add_argument("--smoke", action="store_true", help="REAL-POOL only, 2000 images, tiny reference")
ap.add_argument("--no-save-wrapped", action="store_true")
args = ap.parse_args()

T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arms = [a.strip() for a in args.arms.split(",") if a.strip()]
assert all(a in ARMS for a in arms), arms
cfg_names = []
for tok in args.configs.split(","):
    tok = tok.strip()
    if tok == "all":
        cfg_names += [n for n, _, _ in CONFIGS]
    elif tok == "real":
        cfg_names.append("REAL-POOL")
    elif tok:
        assert tok in M_OF, tok
        cfg_names.append(tok)
NSUB, M_REAL, TAG = args.nsub, args.m_real, args.tag
FD = FEATS
if args.smoke:
    cfg_names, arms, NSUB, M_REAL, TAG, FD = ["REAL-POOL"], ARMS, 1000, 2000, "smoke", f"{FEATS}/smoke"
for d in (FD, RESULTS, f"{RESULTS}/sev", WRAPPED):
    os.makedirs(d, exist_ok=True)
print(f"wrap_and_measure: configs={cfg_names} arms={arms} nsub={NSUB} tag={TAG} device={DEV}", flush=True)

# ------------------------------------------------------------------ layer + splits
L, basis = sc.load_layer()
fit_idx, hold_idx, R50 = L["fit_idx"], L["hold_idx"], L["R50"]
Rf_sorted = np.sort(R50[fit_idx])
n_h = len(hold_idx)
u, xi, sg, zeta, tau3, tau4 = (L[k] for k in ("u", "xi", "sg", "zeta", "tau3", "tau4"))
perm = np.random.default_rng(REAL_SPLIT_SEED).permutation(n_h)
h = hold_idx[perm]
half = n_h // 2
ref_idx, pool_idx = np.sort(h[:half]), np.sort(h[half:])
if args.smoke:
    ref_idx = ref_idx[:20_000]
X8 = np.load(NPY, mmap_mode="r")
print(f"layer: tau3={tau3:.4f} tau4={tau4:.4f} n_h={n_h}  REF={len(ref_idx)} POOL={len(pool_idx)}",
      flush=True)

fe = ft.inception_extractor(DEV)
mae = ft.mae_model(DEV)

# ------------------------------------------------------------------ reference moments (cached)
stats_path = f"{FD}/ref_stats.npz"
inc_path = f"{FD}/inception_ref.npy"
if not os.path.exists(stats_path):
    n = len(ref_idx)
    F_inc = np.empty((n, 2048), np.float32)
    E_mae = np.empty((n, 768), np.float32)
    for a in range(0, n, 100_000):
        Xc = np.ascontiguousarray(X8[ref_idx[a:a + 100_000]])
        F_inc[a:a + len(Xc)] = ft.inception_features(Xc, fe, DEV)
        E_mae[a:a + len(Xc)] = ft.mae_features(Xc, mae, DEV)
        print(f"  [ref feats] {a+len(Xc)}/{n} ({time.time()-T0:.0f}s)", flush=True)
    mu_i, C_i = ft.stats(F_inc)
    mu_m, C_m = ft.stats(E_mae)
    np.save(inc_path, F_inc)
    np.savez(stats_path, mu_inc=mu_i, C_inc=C_i, mu_mae=mu_m, C_mae=C_m, n_ref=n, ref_idx=ref_idx)
    del E_mae
    print(f"reference moments built and cached ({time.time()-T0:.0f}s)", flush=True)
else:
    z = np.load(stats_path)
    mu_i, C_i, mu_m, C_m = z["mu_inc"], z["C_inc"], z["mu_mae"], z["C_mae"]
    assert int(z["n_ref"]) == len(ref_idx), "cached reference does not match this split"
    F_inc = np.load(inc_path)
    print(f"reference moments loaded from cache (n_ref={len(ref_idx)})", flush=True)
F_REF = F_inc            # kept for KID


# ------------------------------------------------------------------ evaluation of one arm
def pixel_stats(X, Xa, chunk=20_000):
    n_mod, l2_sum, l2_max, abs_sum, npix_mod = 0, 0.0, 0.0, 0.0, 0
    for a in range(0, len(X), chunk):
        d = Xa[a:a + chunk].astype(np.int16) - X[a:a + chunk].astype(np.int16)
        d = d.reshape(len(d), -1)
        l2 = np.sqrt((d.astype(np.float64) ** 2).sum(1))
        mod = l2 > 0
        n_mod += int(mod.sum())
        l2_sum += float(l2[mod].sum())
        l2_max = max(l2_max, float(l2.max()) if len(l2) else 0.0)
        abs_sum += float(np.abs(d[mod]).sum())
        npix_mod += int(mod.sum()) * d.shape[1]
    return dict(n_modified=n_mod, frac_modified=n_mod / len(X),
                mean_l2_modified=(l2_sum / n_mod if n_mod else 0.0), max_l2=l2_max,
                mean_abs_pix_modified=(abs_sum / npix_mod if npix_mod else 0.0))


def evaluate(name, arm, X, Xa, info, Rm, Rw, sub):
    t1 = time.time()
    M = len(X)
    R_img = sc.severities(Xa, basis, DEV)
    a3 = sc.audit_rho(R_img, tau3, 1e-3, n_h)
    a4 = sc.audit_rho(R_img, tau4, 1e-4, n_h)
    F = ft.inception_features(Xa, fe, DEV)
    mu, C = ft.stats(F)
    fid_full = ft.frechet(mu, C, mu_i, C_i)
    Fs = F[sub]
    mus, Cs = ft.stats(Fs)
    fid_sub = ft.frechet(mus, Cs, mu_i, C_i)
    kid_m, kid_s = ft.kid(Fs, F_REF)
    del F
    E = ft.mae_features(Xa, mae, DEV)
    mu, C = ft.stats(E)
    fmd_full = ft.frechet(mu, C, mu_m, C_m)
    mu, C = ft.stats(E[sub])
    fmd_sub = ft.frechet(mu, C, mu_m, C_m)
    del E
    row = dict(config=name, arm=arm, M=M, n_sub=len(sub),
               n_transported=info.get("n_transported", 0),
               frac_transported=info.get("n_transported", 0) / M,
               fid_sub=fid_sub, fid_full=fid_full, kid_mean=kid_m, kid_std=kid_s,
               fmd_sub=fmd_sub, fmd_full=fmd_full,
               rho3_raw=float((Rm >= tau3).mean() / 1e-3), rho3_remap=float((Rw >= tau3).mean() / 1e-3),
               rho3_img=a3["rho"], rho3_lo=a3["quad_lo"], rho3_hi=a3["quad_hi"],
               rho3_verdict=a3["verdict"], k3=a3["k"],
               rho4_raw=float((Rm >= tau4).mean() / 1e-4), rho4_remap=float((Rw >= tau4).mean() / 1e-4),
               rho4_img=a4["rho"], rho4_lo=a4["quad_lo"], rho4_hi=a4["quad_hi"],
               rho4_verdict=a4["verdict"], k4=a4["k"])
    row.update(pixel_stats(X, Xa))
    for k in ("oor0", "sev_err_med", "sev_err_p95", "sev_err_max", "sev_within_005"):
        row[k] = info.get(k, np.nan)
    if "transported_idx" in info:
        e8 = np.abs(R_img[info["transported_idx"]] - info["r_target"])
        row.update(sev_err_u8_med=float(np.median(e8)), sev_err_u8_p95=float(np.quantile(e8, .95)),
                   sev_u8_within_005=float((e8 < 0.05).mean()))
    else:
        row.update(sev_err_u8_med=np.nan, sev_err_u8_p95=np.nan, sev_u8_within_005=np.nan)
    row.update(n_ref=len(ref_idx), seconds=round(time.time() - t1, 1))
    np.save(f"{RESULTS}/sev/{TAG}_{SLUG.get(name, 'real_pool')}_{arm}.npy", R_img)
    print(f"  {name:12s} {arm:18s} FID50k={fid_sub:7.3f} FIDfull={fid_full:7.3f} "
          f"KID={kid_m:.5f} FMD50k={fmd_sub:.4f} | rho3 raw={row['rho3_raw']:.2f} "
          f"remap={row['rho3_remap']:.2f} img={row['rho3_img']:.2f} [{a3['quad_lo']:.2f},{a3['quad_hi']:.2f}] "
          f"| rho4 img={row['rho4_img']:.2f} | touched={row['n_modified']} "
          f"meanL2={row['mean_l2_modified']:.1f}  ({time.time()-T0:.0f}s)", flush=True)
    return row


# ------------------------------------------------------------------ csv sink
csv_path = f"{RESULTS}/wrapfid_{TAG}.csv"


def write_row(row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ main loop
for name in cfg_names:
    if name == "REAL-POOL":
        rng = np.random.default_rng(POOL_SEED)
        idx = np.sort(rng.choice(pool_idx, size=min(M_REAL, len(pool_idx)), replace=False))
        X = np.ascontiguousarray(X8[idx])
        slug = "real_pool"
    else:
        slug = SLUG[name]
        X = sc.load_cache(cache_path(slug))
        if len(X) != M_OF[name]:
            print(f"WARNING {name}: cache has {len(X)} != expected M {M_OF[name]}", flush=True)
    M = len(X)
    Rm = sc.severities(X, basis, DEV)
    v, Rw = rank_remap(Rm, Rf_sorted, u, xi, sg, zeta)
    sub = np.sort(np.random.default_rng(SUB_SEED).choice(M, size=min(NSUB, M), replace=False))
    print(f"[{name}] M={M} raw rho3={(Rm >= tau3).mean()/1e-3:.3f} rho4={(Rm >= tau4).mean()/1e-4:.3f} "
          f"| remap rho3={(Rw >= tau3).mean()/1e-3:.3f} rho4={(Rw >= tau4).mean()/1e-4:.3f}  "
          f"({time.time()-T0:.0f}s)", flush=True)
    for arm in arms:
        Xa, info = build_arm(arm, X, Rm, Rw, v, basis, DEV)
        row = evaluate(name, arm, X, Xa, info, Rm, Rw, sub)
        write_row(row)
        if arm != "raw" and not args.no_save_wrapped and not args.smoke:
            np.savez_compressed(f"{WRAPPED}/{TAG}_{slug}_{arm}.npz", X32=Xa,
                                idx=info["transported_idx"], r_target=info["r_target"])
    del X
print(f"wrote {csv_path}\nALL DONE ({time.time()-T0:.0f}s)", flush=True)
