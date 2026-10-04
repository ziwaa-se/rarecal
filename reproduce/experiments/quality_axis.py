"""FMD of the audited samples against a fixed real reference, with the real-versus-real floor."""
import os
import sys
import time
import csv
import numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import analysis_common as E                                        # noqa: E402

t0 = time.time()
os.makedirs(E.P3, exist_ok=True)
REF_CACHE = f"{E.P3}/e7_ref_moments.npz"
SMOKE = os.environ.get("E7_SMOKE", "0") == "1"

ref_idx, pool_idx = E.ref_pool_split()
if SMOKE:
    ref_idx = ref_idx[:40_000]
    pool_idx = pool_idx[:40_000]
print(f"REF n={len(ref_idx)}  POOL n={len(pool_idx)}", flush=True)

Ereal = E.real_emb()

# ---------------- real reference moments -----------------------------------
if os.path.exists(REF_CACHE) and not SMOKE:
    z = np.load(REF_CACHE)
    mu_ref, S_ref = z["mu"], z["S"]
    print("loaded cached REF moments", flush=True)
else:
    mu_ref, S_ref = E.moments(Ereal[ref_idx])
    if not SMOKE:
        np.savez(REF_CACHE, mu=mu_ref, S=S_ref, ref_idx=ref_idx)
    print(f"REF moments built ({time.time()-t0:.0f}s)", flush=True)
C_ref = E.cov_of(mu_ref, S_ref)
print(f"REF: ||mu||={np.linalg.norm(mu_ref):.3f} tr(C)={np.trace(C_ref):.3f}",
      flush=True)

# ---------------- per-config FMD at the audited M ---------------------------
rows_fmd = {}
Ms = {}
for name in E.CONFIG_NAMES:
    Em = E.load_model_emb(name)
    if SMOKE:
        Em = Em[:5000]
    mu, S = E.moments(Em)
    Ms[name] = len(Em)
    rows_fmd[name] = E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref)
    print(f"  {name:12s} M={len(Em):6d} FMD={rows_fmd[name]:8.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del Em

# ---------------- real-vs-real floor at each M used -------------------------
rng = np.random.default_rng(7001)
floor = {}
for M in sorted(set(Ms.values())):
    vals = []
    for rep in range(3):
        sel = np.sort(rng.choice(pool_idx, size=min(M, len(pool_idx)),
                                 replace=False))
        mu, S = E.moments(Ereal[sel])
        vals.append(E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref))
    floor[M] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    print(f"  real-vs-real floor M={M:6d}: {floor[M][0]:.4f} "
          f"+/- {floor[M][1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- FMD at a common budget (bias control) ---------------------
M_COMMON = min(Ms.values())
rng2 = np.random.default_rng(7002)
common = {}
for name in E.CONFIG_NAMES:
    Em = E.load_model_emb(name)
    if SMOKE:
        Em = Em[:5000]
    vals = []
    for rep in range(5):
        sel = rng2.choice(len(Em), size=min(M_COMMON, len(Em)), replace=False)
        mu, S = E.moments(Em[np.sort(sel)])
        vals.append(E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref))
    common[name] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    print(f"  {name:12s} FMD@M={M_COMMON}: {common[name][0]:.4f} "
          f"+/- {common[name][1]:.4f}", flush=True)
    del Em
vals = []
for rep in range(5):
    sel = np.sort(rng2.choice(pool_idx, size=min(M_COMMON, len(pool_idx)),
                              replace=False))
    mu, S = E.moments(Ereal[sel])
    vals.append(E.frechet(mu, E.cov_of(mu, S), mu_ref, C_ref))
floor_common = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
print(f"  real-vs-real floor @M={M_COMMON}: {floor_common[0]:.4f} "
      f"+/- {floor_common[1]:.4f}", flush=True)

# ---------------- join with the frozen R1 leaderboard -----------------------
lb = {}
for r in csv.DictReader(open(f"{E.DATA}/leaderboard_r1.csv")):
    if abs(float(r["p"]) - 1e-3) < 1e-12:
        lb[r["config"]] = r

rows = []
for name in E.CONFIG_NAMES:
    r = lb[name]
    if not SMOKE:
        assert int(r["M"]) == Ms[name], (f"{name}: audited M {r['M']} != "
                                         f"embedding M {Ms[name]}")
    fid = r["fid_published"]
    rows.append(dict(
        config=name, family=r["family"], year=r["year"],
        M=Ms[name],
        fmd_mae_auditedM=round(rows_fmd[name], 6),
        fmd_real_floor_sameM=round(floor[Ms[name]][0], 6),
        fmd_real_floor_sd=round(floor[Ms[name]][1], 6),
        fmd_mae_commonM=round(common[name][0], 6),
        fmd_mae_commonM_sd=round(common[name][1], 6),
        M_common=M_COMMON,
        fid_published=fid,
        # resolutions as reported by each model's authors
        # (sections/experiments.-52)
        fid_resolution={"iDDPM": "64x64", "EDM": "64x64", "EDM-churn": "",
                        "EDM2-S": "64x64", "EDM2-M": "64x64",
                        "StyleGAN-XL": "32x32", "DiT-cfg1.5": "256x256",
                        "DiT-cfg1.0": "256x256"}[name],
        fmd_resolution="32x32",
        rho_s1_1e3=float(r["rho"]), rho_quad_lo=float(r["quad_lo"]),
        rho_quad_hi=float(r["quad_hi"]), k=int(r["k"]),
        verdict=r["verdict_quad"],
        abs_log_rho=round(abs(np.log(float(r["rho"]))), 6)))

E.writecsv(f"{E.DATA}/quality_axis_fmd.csv", rows)

# ---------------- rank correlations -----------------------------------------
def report(xname, x, yname, y, tag):
    sp = spearmanr(x, y)
    kt = kendalltau(x, y)
    print(f"  {tag:44s} n={len(x)}  spearman={sp.statistic:+.4f} "
          f"(p={sp.pvalue:.3f})  kendall={kt.statistic:+.4f} "
          f"(p={kt.pvalue:.3f})", flush=True)
    return dict(tag=tag, x=xname, y=yname, n=len(x),
                spearman=round(float(sp.statistic), 4),
                spearman_p=round(float(sp.pvalue), 4),
                kendall_tau_b=round(float(kt.statistic), 4),
                kendall_p=round(float(kt.pvalue), 4))


print("\nRANK CORRELATIONS", flush=True)
allr = rows
fid_r = [r for r in rows if r["fid_published"] != ""]
corr = []
corr.append(report("fmd_mae_auditedM", [r["fmd_mae_auditedM"] for r in allr],
                   "rho_s1_1e3", [r["rho_s1_1e3"] for r in allr],
                   "FMD (audited M) vs rho"))
corr.append(report("fmd_mae_commonM", [r["fmd_mae_commonM"] for r in allr],
                   "rho_s1_1e3", [r["rho_s1_1e3"] for r in allr],
                   "FMD (common M) vs rho"))
corr.append(report("fmd_mae_auditedM", [r["fmd_mae_auditedM"] for r in allr],
                   "abs_log_rho", [r["abs_log_rho"] for r in allr],
                   "FMD (audited M) vs |log rho|"))
corr.append(report("fid_published", [float(r["fid_published"]) for r in fid_r],
                   "rho_s1_1e3", [r["rho_s1_1e3"] for r in fid_r],
                   "published FID vs rho"))
corr.append(report("fid_published", [float(r["fid_published"]) for r in fid_r],
                   "abs_log_rho", [r["abs_log_rho"] for r in fid_r],
                   "published FID vs |log rho|"))
corr.append(report("fid_published", [float(r["fid_published"]) for r in fid_r],
                   "fmd_mae_auditedM",
                   [r["fmd_mae_auditedM"] for r in fid_r],
                   "published FID vs FMD (agreement of the two axes)"))
E.writecsv(f"{E.DATA}/quality_axis_rank_correlations.csv", corr)

# ---------------- adjacent-model FMD gaps (the yardstick) ---------------
order = sorted(allr, key=lambda r: r["fmd_mae_commonM"])
gaps = [(order[i + 1]["config"], order[i]["config"],
         order[i + 1]["fmd_mae_commonM"] - order[i]["fmd_mae_commonM"])
        for i in range(len(order) - 1)]
print("\nADJACENT-MODEL FMD GAPS (common M, the  yardstick)", flush=True)
for a, b, g in gaps:
    print(f"  {a:12s} - {b:12s} = {g:8.4f}", flush=True)
gapvals = [g for _, _, g in gaps]
print(f"  min adjacent gap = {min(gapvals):.4f}   "
      f"median = {np.median(gapvals):.4f}", flush=True)
np.savez(f"{E.P3}/e7c_yardsticks.npz",
         gaps=np.array(gapvals),
         floor_common=np.array(floor_common),
         fmd_common=np.array([r["fmd_mae_commonM"] for r in allr]),
         names=np.array([r["config"] for r in allr]))
print(f"\nE7c DONE ({time.time()-t0:.0f}s)", flush=True)
