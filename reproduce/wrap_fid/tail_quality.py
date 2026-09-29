"""Per-image change, improved precision and tail KID of the transported samples."""
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
ap.add_argument("--n-bulk", type=int, default=50_000)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_SPLIT_SEED, POOL_SEED, M_REAL = 20260816, 20260914, 100_000
QS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
names = []
for tok in args.configs.split(","):
    names += [n for n, _, _ in CONFIGS] if tok == "all" else (["REAL-POOL"] if tok == "real" else [tok])
if args.smoke:
    names, args.n_bulk = ["DiT-cfg1.0"], 5000
FIGS = f"{RESULTS}/figs"
os.makedirs(FIGS, exist_ok=True)
TAG = "smoke_" if args.smoke else ""

# ------------------------------------------------------------------ layer, reference manifold
L, basis = sc.load_layer()
hold_idx, R50 = L["hold_idx"], L["R50"]
tau3 = L["tau3"]
z = np.load(f"{FEATS}/ref_stats.npz")
ref_idx = z["ref_idx"]
F_ref = np.load(f"{FEATS}/inception_ref.npy", mmap_mode="r")
X8 = np.load(NPY, mmap_mode="r")
R_ref = R50[ref_idx]
tail_pos = np.where(R_ref >= np.quantile(R_ref, 0.99))[0]
bulk_pos = np.sort(np.random.default_rng(20260919).choice(len(ref_idx), args.n_bulk, replace=False))
man_pos = np.union1d(tail_pos, bulk_pos)
is_tail = torch.tensor(np.isin(man_pos, tail_pos), device=DEV)
F_man = torch.tensor(np.asarray(F_ref[man_pos]), device=DEV)
man_tail_corpus_idx = ref_idx[man_pos[np.isin(man_pos, tail_pos)]]        # corpus rows of manifold tail, in order
F_realtail = np.asarray(F_ref[tail_pos])
radii = pc.knn_radii(F_man, k=3)
print(f"manifold: {len(man_pos)} reals ({len(tail_pos)} real-tail, R>={np.quantile(R_ref,0.99):.2f}); "
      f"median k=3 radius {radii.median().item():.2f}  ({time.time()-T0:.0f}s)", flush=True)

fe = ft.inception_extractor(DEV)
lp = pc.lpips_net(DEV)


def feats_t(X):
    return torch.tensor(ft.inception_features(X, fe, DEV), device=DEV)


def q(x, p):
    return float(np.quantile(x, p))


rows = []
for name in names:
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
    W = np.load(f"{WRAPPED}/main_{slug}_wrap-tail.npz")
    Xw, tidx, r_tgt = W["X32"].reshape(-1, 32, 32, 3), W["idx"], W["r_target"]
    keep = np.ones(len(X), bool)
    keep[tidx] = False
    assert np.array_equal(X[keep], Xw[keep]), f"{name}: wrapped cache differs outside the tail region"
    A, B = X[tidx], Xw[tidx]                                   # before / after, same order
    n = len(tidx)
    pm = pc.pair_metrics(A, B, DEV, net=lp)
    changed = pm["l2"] > 0
    sat_a, sat_b = pc.saturation(A), pc.saturation(B)
    Ra, Rb = sc.severities(A, basis, DEV), sc.severities(B, basis, DEV)
    Fa, Fb = feats_t(A), feats_t(B)
    in_a, nn_a, _ = pc.realism(Fa, F_man, radii, is_tail)
    in_b, nn_b, nb_tail = pc.realism(Fb, F_man, radii, is_tail)
    bulk_sel = np.sort(np.random.default_rng(7).choice(np.where(keep)[0], size=n, replace=False))
    in_bulk, nn_bulk, _ = pc.realism(feats_t(X[bulk_sel]), F_man, radii, is_tail)
    ks = min(500, n)
    kid_a, kid_a_sd = pc.kid_small(Fa.cpu().numpy(), F_realtail, ks)
    kid_b, kid_b_sd = pc.kid_small(Fb.cpu().numpy(), F_realtail, ks)
    c = changed
    row = dict(config=name, n_touched=n, n_changed=int(c.sum()),
               sev_before_med=q(Ra, .5), sev_after_med=q(Rb, .5), sev_target_med=q(r_tgt, .5),
               mean_abs_pix_med=q(pm["mean_abs"][c], .5) if c.any() else 0.0,
               mean_abs_pix_p95=q(pm["mean_abs"][c], .95) if c.any() else 0.0,
               psnr_med=q(pm["psnr"][c], .5) if c.any() else np.inf,
               psnr_p05=q(pm["psnr"][c], .05) if c.any() else np.inf,
               ssim_med=q(pm["ssim"], .5), ssim_p05=q(pm["ssim"], .05), ssim_min=float(pm["ssim"].min()),
               lpips_med=q(pm["lpips"], .5), lpips_p95=q(pm["lpips"], .95), lpips_max=float(pm["lpips"].max()),
               sat_before_mean=float(sat_a.mean()), sat_after_mean=float(sat_b.mean()),
               sat_after_p95=q(sat_b, .95), frac_sat_gt10pct_before=float((sat_a > .10).mean()),
               frac_sat_gt10pct_after=float((sat_b > .10).mean()),
               precision_before=float(in_a.mean()), precision_after=float(in_b.mean()),
               precision_bulk_same_model=float(in_bulk.mean()),
               lost_precision=int((in_a & ~in_b).sum()), gained_precision=int((~in_a & in_b).sum()),
               nn_dist_before_med=q(nn_a, .5), nn_dist_after_med=q(nn_b, .5), nn_dist_bulk_med=q(nn_bulk, .5),
               tailkid_before=kid_a, tailkid_before_sd=kid_a_sd, tailkid_after=kid_b, tailkid_after_sd=kid_b_sd,
               seconds=round(time.time() - t1, 1))
    rows.append(row)
    np.savez_compressed(f"{RESULTS}/{TAG}tailquality_perimage_{slug}.npz", idx=tidx, r_target=r_tgt,
                        R_before=Ra, R_after=Rb, inside_before=in_a, inside_after=in_b,
                        nn_before=nn_a, nn_after=nn_b, sat_before=sat_a, sat_after=sat_b, **pm)
    print(f"{name:12s} touched={n} changed={int(c.sum())} | med|dpix|={row['mean_abs_pix_med']:.2f} "
          f"PSNR med={row['psnr_med']:.1f} SSIM med={row['ssim_med']:.3f} p05={row['ssim_p05']:.3f} "
          f"LPIPS med={row['lpips_med']:.4f} p95={row['lpips_p95']:.4f} | sat {row['sat_before_mean']:.3f}->"
          f"{row['sat_after_mean']:.3f} | precision {row['precision_before']:.3f}->{row['precision_after']:.3f} "
          f"(bulk {row['precision_bulk_same_model']:.3f}) | tailKID {kid_a:.4f}->{kid_b:.4f}  "
          f"({time.time()-T0:.0f}s)", flush=True)

    # ---- gallery at fixed displacement quantiles of the CHANGED images (100% = worst case)
    ci = np.where(c)[0] if c.any() else np.arange(n)
    order = ci[np.argsort(pm["l2"][ci])]
    pick = [order[min(len(order) - 1, int(round(p * (len(order) - 1))))] for p in QS]
    diff = np.clip(np.abs(B[pick].astype(np.int16) - A[pick].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    nn_img = np.stack([X8[man_tail_corpus_idx[j]] for j in nb_tail[pick]])
    cols = [f"disp. q{int(p*100)}\nR {Ra[i]:.1f}→{Rb[i]:.1f}\nSSIM {pm['ssim'][i]:.2f}" for p, i in zip(QS, pick)]
    pc.gallery(f"{FIGS}/{TAG}tailgallery_{slug}",
               f"{name}: wrapped tail samples at displacement quantiles of the touched set (q100 = worst case)",
               [("before", A[pick]), ("after (wrapped)", B[pick]), ("|diff| ×4", diff),
                ("nearest real tail", nn_img)], cols)

out = f"{RESULTS}/{TAG}tailquality.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}", flush=True)

# ---- summary figure: precision of the touched set before / after, with bulk and real-tail ceiling
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
lab = [r["config"] for r in rows]
x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(1.1 * len(rows) + 2.2, 3.2))
ax.bar(x - 0.27, [r["precision_bulk_same_model"] for r in rows], 0.26, label="same model, non-tail samples", color="#b9b6ab")
ax.bar(x, [r["precision_before"] for r in rows], 0.26, label="tail samples, before wrap", color="#eb6834")
ax.bar(x + 0.27, [r["precision_after"] for r in rows], 0.26, label="tail samples, after wrap", color="#2a78d6")
ax.set_xticks(x)
ax.set_xticklabels(lab, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("precision (inside real k=3 NN manifold)")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(f"{FIGS}/{TAG}tail_precision.png", dpi=170)
fig.savefig(f"{FIGS}/{TAG}tail_precision.pdf")
print(f"ALL DONE ({time.time()-T0:.0f}s)", flush=True)
