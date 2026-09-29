"""Paired bootstrap of the change in FID between wrapped and raw samples."""
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
import feats as ft                                                              # noqa: E402
import post_core as pc                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="all,real")
ap.add_argument("--B", type=int, default=200)
ap.add_argument("--nsub", type=int, default=50_000)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_SPLIT_SEED, POOL_SEED, M_REAL, BOOT_SEED = 20260816, 20260914, 100_000, 20260920
ARMS = [("wrap-tail", "main"), ("wrap-tail-hardclip", "ablation"), ("wrap-full", "ablation")]
names = []
for tok in args.configs.split(","):
    names += [n for n, _, _ in CONFIGS] if tok == "all" else (["REAL-POOL"] if tok == "real" else [tok])
if args.smoke:
    names, args.B = ["DiT-cfg1.0"], 3
out_csv = f"{RESULTS}/{'smoke_' if args.smoke else ''}bootstrap_fid.csv"
done = set()
if os.path.exists(out_csv):
    done = {r["config"] for r in csv.DictReader(open(out_csv))}

z = np.load(f"{FEATS}/ref_stats.npz")
ref = pc.FrechetRef(z["mu_inc"], z["C_inc"], DEV)
L, basis = sc.load_layer()
hold_idx = L["hold_idx"]
X8 = np.load(NPY, mmap_mode="r")
fe = ft.inception_extractor(DEV)
print(f"bootstrap: configs={names} B={args.B} n<= {args.nsub} device={DEV} (skip done: {sorted(done)})", flush=True)

for name in names:
    if name in done:
        continue
    t1 = time.time()
    if name == "REAL-POOL":
        n_h = len(hold_idx)
        h = hold_idx[np.random.default_rng(REAL_SPLIT_SEED).permutation(n_h)]
        pool_idx = np.sort(h[n_h // 2:])
        idx = np.sort(np.random.default_rng(POOL_SEED).choice(pool_idx, size=M_REAL, replace=False))
        X, slug = np.ascontiguousarray(X8[idx]), "real_pool"
    else:
        slug = SLUG[name]
        X = sc.load_cache(cache_path(slug))
    M = len(X)
    F = {}
    p_raw = f"{FEATS}/inc_{slug}_raw.npy"
    if os.path.exists(p_raw):
        F["raw"] = np.load(p_raw)
    else:
        F["raw"] = ft.inception_features(X, fe, DEV)
        if not args.smoke:
            np.save(p_raw, F["raw"])
    for arm, tag in ARMS:
        W = np.load(f"{WRAPPED}/{tag}_{slug}_{arm}.npz")
        Xw, tidx = W["X32"].reshape(-1, 32, 32, 3), W["idx"]
        if arm == "wrap-full":
            p_full = f"{FEATS}/inc_{slug}_wrap-full.npy"
            if os.path.exists(p_full):
                F[arm] = np.load(p_full)
            else:
                F[arm] = ft.inception_features(Xw, fe, DEV)
                if not args.smoke:
                    np.save(p_full, F[arm])
        else:                                                   # only the touched rows differ from raw
            Fa = F["raw"].copy()
            Fa[tidx] = ft.inception_features(Xw[tidx], fe, DEV)
            F[arm] = Fa
        del W, Xw
    print(f"[{name}] features ready M={M} ({time.time()-t1:.0f}s)", flush=True)

    Ft = {k: torch.tensor(v, device=DEV) for k, v in F.items()}             # float32 on GPU
    n = min(args.nsub, M)
    g = torch.Generator(device=DEV)
    g.manual_seed(BOOT_SEED)
    draws = {k: np.empty(args.B) for k in Ft}
    for b in range(args.B):
        ib = torch.randint(M, (n,), device=DEV, generator=g)
        for k in Ft:
            draws[k][b] = ref.fid(Ft[k][ib])
        if b % 25 == 0:
            print(f"    draw {b}/{args.B} raw={draws['raw'][b]:.4f} "
                  f"d_tail={draws['wrap-tail'][b]-draws['raw'][b]:+.5f}  ({time.time()-t1:.0f}s)", flush=True)
    point = {k: ref.fid(Ft[k]) for k in Ft}                                  # full-M point estimates
    rows = []
    for arm, _ in ARMS:
        d = draws[arm] - draws["raw"]
        rows.append(dict(config=name, arm=arm, M=M, n=n, B=args.B,
                         fid_raw_fullM=point["raw"], dfid_fullM=point[arm] - point["raw"],
                         fid_raw_boot_mean=float(draws["raw"].mean()), fid_raw_boot_sd=float(draws["raw"].std(ddof=1)),
                         dfid_boot_mean=float(d.mean()), dfid_boot_sd=float(d.std(ddof=1)),
                         dfid_ci_lo=float(np.quantile(d, .025)), dfid_ci_hi=float(np.quantile(d, .975)),
                         frac_positive=float((d > 0).mean()),
                         ratio_absdelta_to_rawsd=float(abs(d.mean()) / draws["raw"].std(ddof=1))))
        r = rows[-1]
        print(f"  {name:12s} {arm:18s} dFID={r['dfid_boot_mean']:+.5f} sd={r['dfid_boot_sd']:.5f} "
              f"95%[{r['dfid_ci_lo']:+.5f},{r['dfid_ci_hi']:+.5f}] | raw FID sd={r['fid_raw_boot_sd']:.4f} "
              f"({time.time()-T0:.0f}s)", flush=True)
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    if not args.smoke:
        np.savez_compressed(f"{RESULTS}/bootstrap_fid_draws_{slug}.npz", **draws)
    del Ft, F, X
    torch.cuda.empty_cache()
print(f"wrote {out_csv}\nALL DONE ({time.time()-T0:.0f}s)", flush=True)
