"""Optional per-sample guard: revert transported images with a large severity residual or low SSIM."""
import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import NPY, CONFIGS, SLUG, FEATS, RESULTS, WRAPPED, cache_path      # noqa: E402
import severity as sc                                                           # noqa: E402
from wrap_core import rank_remap, transport, quantize                           # noqa: E402
import feats as ft                                                              # noqa: E402
import post_core as pc                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="all,real")
ap.add_argument("--B", type=int, default=200)
ap.add_argument("--res-tol", type=float, default=0.05)
ap.add_argument("--ssim-tol", type=float, default=0.80)
args = ap.parse_args()
T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_SPLIT_SEED, POOL_SEED, SUB_SEED, BOOT_SEED, M_REAL, NSUB = 20260816, 20260914, 20260915, 20260920, 100_000, 50_000
OUT = f"{RESULTS}/guard"
os.makedirs(f"{OUT}/figs", exist_ok=True)
names = []
for tok in args.configs.split(","):
    names += [n for n, _, _ in CONFIGS] if tok == "all" else (["REAL-POOL"] if tok == "real" else [tok])

# ------------------------------------------------------------------ frozen inputs
L, basis = sc.load_layer()
fit_idx, hold_idx, R50 = L["fit_idx"], L["hold_idx"], L["R50"]
Rf_sorted = np.sort(R50[fit_idx])
n_h = len(hold_idx)
u, xi, sg, zeta, tau3, tau4 = (L[k] for k in ("u", "xi", "sg", "zeta", "tau3", "tau4"))
z = np.load(f"{FEATS}/ref_stats.npz")
mu_i, C_i, mu_m, C_m, ref_idx = z["mu_inc"], z["C_inc"], z["mu_mae"], z["C_mae"], z["ref_idx"]
F_REF = np.load(f"{FEATS}/inception_ref.npy")
ref = pc.FrechetRef(mu_i, C_i, DEV)
X8 = np.load(NPY, mmap_mode="r")
fe = ft.inception_extractor(DEV)
mae = ft.mae_model(DEV)

# reference manifold for precision, identical to tail_quality.py
R_ref = R50[ref_idx]
tail_pos = np.where(R_ref >= np.quantile(R_ref, 0.99))[0]
bulk_pos = np.sort(np.random.default_rng(20260919).choice(len(ref_idx), 50_000, replace=False))
man_pos = np.union1d(tail_pos, bulk_pos)
is_tail = torch.tensor(np.isin(man_pos, tail_pos), device=DEV)
F_man = torch.tensor(F_REF[man_pos], device=DEV)
man_tail_corpus_idx = ref_idx[man_pos[np.isin(man_pos, tail_pos)]]
radii = pc.knn_radii(F_man, k=3)
print(f"guard: configs={names} res_tol={args.res_tol} ssim_tol={args.ssim_tol} B={args.B}  "
      f"({time.time()-T0:.0f}s)", flush=True)


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


for name in names:
    t1 = time.time()
    if name == "REAL-POOL":
        h = hold_idx[np.random.default_rng(REAL_SPLIT_SEED).permutation(n_h)]
        pool_idx = np.sort(h[n_h // 2:])
        X = np.ascontiguousarray(X8[np.sort(np.random.default_rng(POOL_SEED).choice(pool_idx, size=M_REAL, replace=False))])
        slug = "real_pool"
    else:
        slug = SLUG[name]
        X = sc.load_cache(cache_path(slug))
    M = len(X)
    Rm = sc.severities(X, basis, DEV)
    v, Rw = rank_remap(Rm, Rf_sorted, u, xi, sg, zeta)
    idx = np.where(v > 0.99)[0]
    Xt, r_fin, oor0 = transport(X[idx], Rw[idx], basis, DEV, mode="damped")
    Xq = quantize(Xt).reshape(-1, 32, 32, 3)
    res = np.abs(r_fin - Rw[idx])
    A = X[idx]
    pm = pc.pair_metrics(A, Xq, DEV, net=None)
    # determinism check against the stored unguarded wrapper (read-only)
    old = np.load(f"{WRAPPED}/main_{slug}_wrap-tail.npz")["X32"].reshape(-1, 32, 32, 3)[idx]
    n_diff_old = int((old.reshape(len(idx), -1) != Xq.reshape(len(idx), -1)).any(1).sum())
    revert = {"wrap-tail": np.zeros(len(idx), bool),
              "guard-res": res > args.res_tol,
              "guard-res+ssim": (res > args.res_tol) | (pm["ssim"] < args.ssim_tol)}

    # features of the raw set (saved by bootstrap_fid.py, read-only) and of the transported rows
    F_raw = np.load(f"{FEATS}/inc_{slug}_raw.npy")
    F_tr = ft.inception_features(Xq, fe, DEV)
    E_raw = ft.mae_features(X, mae, DEV)
    E_tr = ft.mae_features(Xq, mae, DEV)
    sub = np.sort(np.random.default_rng(SUB_SEED).choice(M, size=min(NSUB, M), replace=False))
    F_raw_t = torch.tensor(F_raw, device=DEV)
    g = torch.Generator(device=DEV)
    g.manual_seed(BOOT_SEED)
    boot_idx = [torch.randint(M, (min(NSUB, M),), device=DEV, generator=g) for _ in range(args.B)]
    boot_raw = np.array([ref.fid(F_raw_t[ib]) for ib in boot_idx])
    fid_raw_sub = ft.frechet(*ft.stats(F_raw[sub]), mu_i, C_i)
    in_before, _, _ = pc.realism(torch.tensor(F_raw[idx], device=DEV), F_man, radii, is_tail)
    print(f"[{name}] M={M} touched={len(idx)} non-converged={int((res > args.res_tol).sum())} "
          f"SSIM<{args.ssim_tol}={int((pm['ssim'] < args.ssim_tol).sum())} "
          f"rows differing from stored unguarded cache={n_diff_old}  ({time.time()-t1:.0f}s)", flush=True)

    for variant, rv in revert.items():
        keep = ~rv
        Xa = X.copy()
        Xa[idx[keep]] = Xq[keep]
        F = F_raw.copy()
        F[idx[keep]] = F_tr[keep]
        E = E_raw.copy()
        E[idx[keep]] = E_tr[keep]
        R_img = sc.severities(Xa, basis, DEV)
        a3, a4 = sc.audit_rho(R_img, tau3, 1e-3, n_h), sc.audit_rho(R_img, tau4, 1e-4, n_h)
        fid_sub = ft.frechet(*ft.stats(F[sub]), mu_i, C_i)
        fid_full = ft.frechet(*ft.stats(F), mu_i, C_i)
        kid_m, kid_s = ft.kid(F[sub], F_REF)
        fmd_sub = ft.frechet(*ft.stats(E[sub]), mu_m, C_m)
        Ft = torch.tensor(F, device=DEV)
        d = np.array([ref.fid(Ft[ib]) for ib in boot_idx]) - boot_raw
        in_after, _, nb_tail = pc.realism(Ft[idx], F_man, radii, is_tail)
        B_img = Xa[idx]
        ss = np.where(rv, 1.0, pm["ssim"])
        e8 = np.abs(R_img[idx] - Rw[idx])
        row = dict(config=name, variant=variant, M=M, n_touched=len(idx), n_reverted=int(rv.sum()),
                   fid_raw_sub=fid_raw_sub, fid_sub=fid_sub, dfid_sub=fid_sub - fid_raw_sub, fid_full=fid_full,
                   kid_mean=kid_m, kid_std=kid_s, fmd_sub=fmd_sub,
                   dfid_boot_mean=float(d.mean()), dfid_boot_sd=float(d.std(ddof=1)),
                   dfid_ci_lo=float(np.quantile(d, .025)), dfid_ci_hi=float(np.quantile(d, .975)),
                   fid_raw_boot_sd=float(boot_raw.std(ddof=1)),
                   rho3_raw=float((Rm >= tau3).mean() / 1e-3), rho3_remap=float((Rw >= tau3).mean() / 1e-3),
                   rho3_img=a3["rho"], rho3_lo=a3["quad_lo"], rho3_hi=a3["quad_hi"], rho3_verdict=a3["verdict"], k3=a3["k"],
                   rho4_img=a4["rho"], rho4_lo=a4["quad_lo"], rho4_hi=a4["quad_hi"], rho4_verdict=a4["verdict"], k4=a4["k"],
                   ssim_min=float(ss.min()), n_ssim_lt_080=int((ss < 0.80).sum()), n_ssim_lt_090=int((ss < 0.90).sum()),
                   n_residual_gt_005_u8=int(((e8 > 0.05) & keep).sum()),
                   precision_before=float(in_before.mean()), precision_after=float(in_after.mean()),
                   sat_before=float(pc.saturation(A).mean()), sat_after=float(pc.saturation(B_img).mean()))
        append(f"{OUT}/guard_results.csv", row)
        print(f"  {name:12s} {variant:15s} reverted={row['n_reverted']:3d} | FID50k {fid_raw_sub:.4f}->{fid_sub:.4f} "
              f"boot dFID={row['dfid_boot_mean']:+.5f} [{row['dfid_ci_lo']:+.5f},{row['dfid_ci_hi']:+.5f}] | "
              f"rho3 {row['rho3_raw']:.2f}->{a3['rho']:.2f} [{a3['quad_lo']:.2f},{a3['quad_hi']:.2f}] {a3['verdict']} "
              f"rho4 {a4['rho']:.2f} | SSIM min={row['ssim_min']:.2f} n<.8={row['n_ssim_lt_080']} | "
              f"precision {row['precision_before']:.3f}->{row['precision_after']:.3f}  ({time.time()-T0:.0f}s)", flush=True)
        if variant != "wrap-tail":
            np.savez_compressed(f"{WRAPPED}/guard_{slug}_{variant}.npz", X32=Xa, idx=idx, r_target=Rw[idx], reverted=rv)
        # gallery of the worst remaining cases (largest displacement among kept samples)
        if variant != "wrap-tail" or name == "iDDPM":
            order = np.argsort(-np.where(keep, pm["l2"], -1.0))[:8]
            diff = np.clip(np.abs(B_img[order].astype(np.int16) - A[order].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            nn_img = np.stack([X8[man_tail_corpus_idx[j]] for j in nb_tail[order]])
            cols = [f"#{k+1} largest disp.\nR {Rm[idx][i]:.1f}→{R_img[idx][i]:.1f}\nSSIM {ss[i]:.2f}" for k, i in enumerate(order)]
            pc.gallery(f"{OUT}/figs/worst8_{slug}_{variant}",
                       f"{name} [{variant}]: the 8 most-displaced wrapped samples that are KEPT",
                       [("before", A[order]), ("after", B_img[order]), ("|diff| ×4", diff), ("nearest real tail", nn_img)], cols)
    del X, F_raw, F_raw_t, E_raw
    torch.cuda.empty_cache()
print(f"wrote {OUT}/guard_results.csv\nALL DONE ({time.time()-T0:.0f}s)", flush=True)
