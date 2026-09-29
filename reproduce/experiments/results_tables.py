"""Assemble the result tables (tail ratios, intervals, verdicts) from the audit outputs."""
import os
import sys
import csv
import shutil
import numpy as np
from scipy.stats import kendalltau
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

VR = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
DATA = os.path.expandvars("${PROJECT_ROOT}/paper/data")
os.makedirs(DATA, exist_ok=True)
Z = 1.96

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, N = sc.split_indices(8000)
NH = len(hold_idx)
print(f"n_h = {NH}", flush=True)

META = {  # family, year, published FID (as reported by each model's authors)
    "iDDPM":       ("diffusion", 2021, "2.9"),
    "EDM":         ("diffusion (Heun)", 2022, "2.6"),
    "EDM-churn":   ("diffusion (same net, stochastic sampler)", 2022, ""),
    "EDM2-S":      ("diffusion", 2024, "1.58"),
    "EDM2-M":      ("diffusion", 2024, "1.43"),
    "StyleGAN-XL": ("GAN", 2022, "1.10"),
    "DiT-cfg1.5":  ("latent diffusion (cfg 1.5)", 2023, "2.27"),
    "DiT-cfg1.0":  ("latent diffusion (same net, no guidance)", 2023, "9.6"),
}
FINAL_SLUG = {"DiT-cfg1.5": "dit_cfg15", "DiT-cfg1.0": "dit_cfg10"}


def r50_of(name):
    slug = sc.SLUG[name]
    if name in FINAL_SLUG:
        W = np.load(f"{sc.P2}/model_w100_{slug}_final.npz")["W"]
    else:
        W = np.load(f"{sc.P2}/model_w100_{slug}.npz")["W"]
    return sc.radius_k(W, 50)


def r_func(name, prefix):
    slug = sc.SLUG[name]
    suf = "_final" if name in FINAL_SLUG else ""
    return np.load(f"{sc.P2}/{prefix}_{slug}{suf}.npy")


def quad(rho, lo_w, hi_w, p, nh=NH, z=Z):
    sig = np.sqrt((1 - p) / (nh * p))
    add = z * sig * rho
    return (rho - np.sqrt((rho - lo_w) ** 2 + add ** 2),
            rho + np.sqrt((hi_w - rho) ** 2 + add ** 2))


def writecsv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.basename(path)} ({len(rows)} rows)", flush=True)


# ---------------- leaderboard_r1.csv ----------------------------------------
tau1 = {1e-3: float(L["tau3"]), 1e-4: float(L["tau4"])}
rows = []
R50_by = {}
for name in META:
    Rm = r50_of(name)
    R50_by[name] = Rm
    for p in [1e-3, 1e-4]:
        a = sc.audit_rho(Rm, tau1[p], p)
        qlo, qhi = quad(a["rho"], a["lo"], a["hi"], p)
        if a["k"] <= 1:
            cl, ch = sc.clopper_pearson(a["k"], a["M"])
            cp_lo, cp_hi = cl / p, ch / p
        else:
            cp_lo = cp_hi = ""
        fam, yr, fid = META[name]
        rows.append(dict(config=name, family=fam, year=yr, fid_published=fid,
                         p=p, M=a["M"], threshold=tau1[p], k=a["k"],
                         rho=a["rho"], wilson_lo=a["lo"], wilson_hi=a["hi"],
                         quad_lo=qlo, quad_hi=qhi,
                         clopper_lo=cp_lo, clopper_hi=cp_hi,
                         verdict_wilson=sc.verdict(a["lo"], a["hi"]),
                         verdict_quad=sc.verdict(qlo, qhi)))
writecsv(f"{DATA}/leaderboard_r1.csv", rows)

# ---------------- leaderboard_functional.csv + concordance ------------------
R2h = np.load(f"{sc.P2}/r2_reals.npy")[hold_idx]
R3h = np.load(f"{sc.P2}/r3_reals.npy")[hold_idx]
taus = {"S1": tau1[1e-3],
        "S2": np.quantile(R2h, 1 - 1e-3),
        "S3": np.quantile(R3h, 1 - 1e-3)}
frows = []
rhos_new = {"S1": [], "S2": [], "S3": []}
verd_new = {"S1": [], "S2": [], "S3": []}
for name in META:
    for func in ["S1", "S2", "S3"]:
        r = R50_by[name] if func == "S1" else r_func(name, func.lower().replace("s", "r"))
        a = sc.audit_rho(r, taus[func], 1e-3)
        qlo, qhi = quad(a["rho"], a["lo"], a["hi"], 1e-3)
        v = sc.verdict(a["lo"], a["hi"])
        note = ""
        if name == "EDM-churn":
            note = "small-M (paused cache M=15,104)"
        if name in FINAL_SLUG:
            note = "final from FINAL 50k cache"
        frows.append(dict(config=name, functional=func, M=a["M"], k=a["k"],
                          rho=a["rho"], wilson_lo=a["lo"], wilson_hi=a["hi"],
                          quad_lo=qlo, quad_hi=qhi, verdict=v, note=note))
        rhos_new[func].append(a["rho"])
        verd_new[func].append(v)
writecsv(f"{DATA}/leaderboard_functional.csv", frows)

# concordance: recomputed with final DiT; partial values recomputed
# from functional_concordance.csv for the old-vs-new comparison
old = list(csv.DictReader(open(f"{VR}/functional_concordance.csv")))
rhos_old = {f: [float(r[f"rho_{f}"]) for r in old] for f in ["S1", "S2", "S3"]}
verd_old = {f: [r[f"verdict_{f}"] for r in old] for f in ["S1", "S2", "S3"]}
crows = []
for a_, b_ in [("S1", "S2"), ("S1", "S3"), ("S2", "S3")]:
    agree = sum(x == y for x, y in zip(verd_new[a_], verd_new[b_]))
    tb = kendalltau(rhos_new[a_], rhos_new[b_]).statistic
    agree_o = sum(x == y for x, y in zip(verd_old[a_], verd_old[b_]))
    tb_o = kendalltau(rhos_old[a_], rhos_old[b_]).statistic
    crows.append(dict(pair=f"{a_}-{b_}", n_configs=8,
                      verdict_agreement=agree, kendall_tau_b=round(tb, 4),
                      verdict_agreement_earlier_samples=agree_o,
                      kendall_tau_b_earlier_samples=round(tb_o, 4)))
    print(f"{a_}-{b_}: agree {agree}/8 (old {agree_o}/8), "
          f"tau_b {tb:+.3f} (old {tb_o:+.3f})", flush=True)
writecsv(f"{DATA}/concordance_stats.csv", crows)

# ---------------- self_validation.csv (+ quadrature accounting) -------------
sv = list(csv.DictReader(open(f"{VR}/self_validation.csv")))
srows = []
for r in sv:
    p = float(r["p"]); rho = float(r["rho_self"])
    lo_w, hi_w = float(r["lo"]), float(r["hi"])
    qlo, qhi = quad(rho, lo_w, hi_w, p)
    srows.append(dict(seed=r["seed"], p=p, tau=r["tau"], k=r["k"], M=r["M"],
                      rho_self=rho, wilson_lo=lo_w, wilson_hi=hi_w,
                      covers_1_wilson=r["covers_1"],
                      quad_lo=qlo, quad_hi=qhi,
                      covers_1_quad=int(qlo <= 1.0 <= qhi)))
writecsv(f"{DATA}/self_validation.csv", srows)
cw = sum(int(r["covers_1_wilson"]) for r in srows)
cq = sum(r["covers_1_quad"] for r in srows)
print(f"self-validation coverage: Wilson {cw}/15, quadrature {cq}/15", flush=True)

# ---------------- self_error.csv (corrected accounting) ---------------------
se = list(csv.DictReader(open(f"{VR}/self_error.csv")))
erows = [dict(seed=r["seed"], p=r["p"],
              e_inherited_continuous=r["e_inherited_continuous"],
              e_rankgrid_M100k=r["e_wrapped_rankgrid"],
              e_forward=r["e_forward"],
              Fhat_inv_threshold=r["Fhat_inv_thr"]) for r in se]
writecsv(f"{DATA}/self_error.csv", erows)

# ---------------- straight consolidations -----------------------------------
for src, dst in [("self_validation_resplit.csv", "self_validation_resplit.csv"),
                 ("planted_ratio.csv", "planted_ratio.csv"),
                 ("planted_ratio_coverage.csv", "planted_ratio_coverage.csv"),
                 ("wrap_end_to_end.csv", "wrap_end_to_end.csv"),
                 ("wrapped_independence.csv", "wrapped_independence.csv"),
                 ("shape_collapse.csv", "shape_collapse.csv"),
                 ("downstream_inheritance.csv", "downstream_inheritance.csv"),
                 ("sensitivity.csv", "sensitivity.csv"),
                 ("sensitivity_gpd.csv", "sensitivity_gpd.csv"),
                 ("scale_curves.csv", "scale_curves.csv"),
                 ("ablation.csv", "ablation.csv")]:
    # FINAL (2026-08-16): these three were recomputed after —
    # shape_collapse at final DiT budgets with deterministic seeding (28/32,
    # not the 25/32), scale_curves and wrapped_independence at the
    # final M=50,000 DiT caches. Copying the originals over them
    # silently reverts published numbers, so they are skipped here.
    if dst in ("shape_collapse.csv", "scale_curves.csv",
               "wrapped_independence.csv"):
        print(f"SKIPPED {dst} (final copy is authoritative)", flush=True)
        continue
    shutil.copyfile(f"{VR}/{src}", f"{DATA}/{dst}")
    print(f"copied {dst}", flush=True)

# ---------------- depth_scan.csv ----------------------------------------------
Rh50 = L["R50"][hold_idx]
levels = np.logspace(np.log10(1e-2), np.log10(1e-4), 10)
drows = []
for name in META:
    Rm = R50_by[name]
    for p in levels:
        t = np.quantile(Rh50, 1 - p)
        a = sc.audit_rho(Rm, t, p)
        qlo, qhi = quad(a["rho"], a["lo"], a["hi"], p)
        drows.append(dict(config=name, p=p, threshold=t, M=a["M"], k=a["k"],
                          rho=a["rho"], wilson_lo=a["lo"], wilson_hi=a["hi"],
                          quad_lo=qlo, quad_hi=qhi))
writecsv(f"{DATA}/depth_scan.csv", drows)

# ---------------- comparator gate -------------------------------------------
lb = list(csv.DictReader(open(f"{DATA}/leaderboard_r1.csv")))
c1 = all("fid_published" in r for r in lb) and any(r["fid_published"] for r in lb)
ab = list(csv.DictReader(open(f"{DATA}/ablation.csv")))
c7 = (any(r["component"] == "gpd_wrapper" for r in ab)
      and any(r["component"] == "transport_chart" for r in ab)
      and any(r["component"] == "gpd_audit_threshold" for r in ab))
scv = list(csv.DictReader(open(f"{DATA}/scale_curves.csv")))
c9 = any(r["M_sub"] == "50000" for r in scv)
print(f"COMPARATOR GATE: C1 fid_published={c1}  C7 ablation rows={c7}  "
      f"budget-50k rows={c9}", flush=True)
assert c1 and c7 and c9
print("ALL CSVS DONE", flush=True)
