"""Conditional tail dispersion test on the calibration-tail PC basis (Brown-Forsythe and permutation tests, Holm correction)."""
import os
import sys
import time
import numpy as np
from scipy.stats import levene
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import analysis_common as E                                        # noqa: E402
import severity as sc                                        # noqa: E402

t0 = time.time()
SMOKE = os.environ.get("E7_SMOKE", "0") == "1"
SEEDS = [20260816, 987654321]
NPERM = 10_000 if not SMOKE else 500
NBOOT = 2_000 if not SMOKE else 200
DEPTHS = [1e-3, 1e-2]
PCS = ["PC1", "PC2"]

fit_idx, hold_idx, _ = sc.split_indices(8000)
Wall = np.load(f"{sc.P2}/wall100_seed8000.npy", mmap_mode="r")
W50 = np.asarray(Wall[:, :50])
R50 = np.sqrt((W50.astype(np.float64) ** 2).sum(1))
Rf, Rh = R50[fit_idx], R50[hold_idx]

# fixed fit-tail PC basis (same basis as the audit)
mB = Rf >= np.quantile(Rf, 0.98)
Zdir = W50[fit_idx][mB] / Rf[mB][:, None]
mZ, sZ = Zdir.mean(0), Zdir.std(0)
pca2 = np.linalg.svd((Zdir - mZ) / sZ, full_matrices=False)[2][:2]
print(f"fit-tail PC basis built ({time.time()-t0:.0f}s)", flush=True)


def proj2(Wrows, Rrows):
    Z = Wrows / Rrows[:, None]
    return ((Z - mZ) / sZ) @ pca2.T


def perm_pval(x, y, nperm, rng):
    obs = abs(np.log(x.var(ddof=1) / y.var(ddof=1)))
    pool = np.concatenate([x, y])
    nx = len(x)
    cnt = 0
    chunk = 500
    for a in range(0, nperm, chunk):
        nb = min(chunk, nperm - a)
        idx = np.argsort(rng.random((nb, len(pool))), axis=1)
        px = pool[idx[:, :nx]]
        py = pool[idx[:, nx:]]
        stat = np.abs(np.log(px.var(axis=1, ddof=1) / py.var(axis=1, ddof=1)))
        cnt += int((stat >= obs).sum())
    return (cnt + 1) / (nperm + 1)


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


# cache the projections once; only the RNG changes between seeds
cells = []
for ic, name in enumerate(E.CONFIG_NAMES):
    Wm = np.load(E.w100_path(name))["W"][:, :50]
    Rm = np.sqrt((Wm.astype(np.float64) ** 2).sum(1))
    M = len(Rm)
    for iq, q in enumerate(DEPTHS):
        n_g = int(np.ceil(q * M))
        gsel = np.argsort(Rm)[-n_g:]
        hsel = hold_idx[Rh >= np.quantile(Rh, 1 - q)]
        Pg = proj2(Wm[gsel], Rm[gsel])
        Ph = proj2(W50[hsel], R50[hsel])
        for ip, pc in enumerate(PCS):
            cells.append(dict(config=name, ic=ic, M=M, depth=q, iq=iq, pc=pc,
                              ip=ip, x=Pg[:, ip].copy(), y=Ph[:, ip].copy(),
                              sample_set=("final" if name in E.FINAL_SLUG
                                     else "single")))
    print(f"  {name:12s} M={M:6d} projections ready ({time.time()-t0:.0f}s)",
          flush=True)

rows = []
summary = {}
for seed in SEEDS:
    blk = []
    for cl in cells:
        x, y = cl["x"], cl["y"]
        rng = np.random.default_rng([seed, cl["ic"], cl["iq"], cl["ip"]])
        bf_p = levene(x, y, center="median").pvalue
        pm_p = perm_pval(x, y, NPERM, rng)
        ratio = x.std(ddof=1) / y.std(ddof=1)
        brng = np.random.default_rng([seed, cl["ic"], cl["iq"], cl["ip"], 7])
        bs = np.empty(NBOOT)
        for i in range(NBOOT):
            bs[i] = (x[brng.integers(0, len(x), len(x))].std(ddof=1)
                     / y[brng.integers(0, len(y), len(y))].std(ddof=1))
        blk.append(dict(seed=seed, config=cl["config"], M=cl["M"],
                        cache=cl["cache"], depth=cl["depth"], pc=cl["pc"],
                        n_g=len(x), n_h=len(y), sd_gen=x.std(ddof=1),
                        sd_real=y.std(ddof=1), sd_ratio=ratio,
                        ratio_lo=np.quantile(bs, 0.025),
                        ratio_hi=np.quantile(bs, 0.975),
                        p_bf=bf_p, p_perm=pm_p))
    hp = holm([r["p_perm"] for r in blk])
    for r, h in zip(blk, hp):
        r["p_holm"] = h
        r["reject_05"] = int(h < 0.05)
    nrej = sum(r["reject_05"] for r in blk)
    summary[seed] = (nrej, blk)
    rows.extend(blk)
    pc1 = [r["sd_ratio"] for r in blk if r["pc"] == "PC1"]
    print(f"\nseed {seed}: {nrej}/{len(blk)} cells reject at Holm 0.05  "
          f"| PC1 sd_ratio range {min(pc1):.4f}-{max(pc1):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    for r in blk:
        if not r["reject_05"]:
            print(f"    ns: {r['config']:12s} q={r['depth']:g} {r['pc']} "
                  f"n_g={r['n_g']:4d} ratio={r['sd_ratio']:.3f} "
                  f"p_perm={r['p_perm']:.4f} p_holm={r['p_holm']:.4f}",
                  flush=True)

E.writecsv(f"{E.DATA}/shape_dispersion.csv", rows)

# ---------------- stability across seeds + delta vs the reported CSV ---------
import csv
a, b = SEEDS
ra_ = {(r["config"], r["depth"], r["pc"]): r for r in summary[a][1]}
rb_ = {(r["config"], r["depth"], r["pc"]): r for r in summary[b][1]}
flips = [k for k in ra_ if ra_[k]["reject_05"] != rb_[k]["reject_05"]]
print(f"\nseed-to-seed verdict flips: {len(flips)}  {flips}", flush=True)

old = {}
for r in csv.DictReader(open(f"{E.DATA}/shape_collapse.csv")):
    old[(r["config"], float(r["depth"]), r["pc"])] = r
print("\nCELLS WHOSE VERDICT CHANGES vs the reported shape_collapse.csv "
      "(hash-seeded, partial DiT):", flush=True)
nchg = 0
for k, r in ra_.items():
    o = old[k]
    if int(o["reject_05"]) != r["reject_05"]:
        nchg += 1
        print(f"  {k[0]:12s} q={k[1]:g} {k[2]}: n_g {o['n_g']}->{r['n_g']}  "
              f"p_holm {float(o['p_holm']):.4f}->{r['p_holm']:.4f}  "
              f"{'ns->REJECT' if r['reject_05'] else 'REJECT->ns'}", flush=True)
print(f"  ({nchg} of 32 change)", flush=True)
print(f"\nrejections: {summary[a][0]}/32 (seed {a}), {summary[b][0]}/32 "
      f"(seed {b}); reported value was 25/32", flush=True)
print(f" DONE ({time.time()-t0:.0f}s)", flush=True)
