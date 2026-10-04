"""Point estimate and interval width of the audit as a function of the sample budget M'."""
import os
import sys
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/paper/data")
rng = np.random.default_rng(42)

L = np.load(f"{sc.P2}/layer_seed8000.npz")
tau3, tau4 = float(L["tau3"]), float(L["tau4"])
GRID = [5000, 10_000, 20_000, 50_000, 100_000]
B = 200
rows = []
for name, _ in sc.CONFIGS:
    suf = "_final" if name in ("DiT-cfg1.5", "DiT-cfg1.0") else ""
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[name]}{suf}.npz")["W"], 50)
    M = len(Rm)
    for p, tau in [(1e-3, tau3), (1e-4, tau4)]:
        full_k = int((Rm >= tau).sum())
        full_rho = full_k / (M * p)
        for Mp in GRID:
            if Mp > M:
                continue
            if Mp == M:
                lo, hi = sc.wilson(full_k, M)
                rows.append(dict(config=name, p=p, M_sub=Mp, B=1,
                                 mean_rho=full_rho, sd_rho=0.0,
                                 mean_ci_width=(hi - lo) / p,
                                 frac_ci_covers_full=1.0, full_rho=full_rho))
                continue
            ks = np.empty(B, int)
            wid = np.empty(B)
            cov = np.zeros(B, bool)
            for b in range(B):
                sub = Rm[rng.choice(M, Mp, replace=False)]
                ks[b] = int((sub >= tau).sum())
                lo, hi = sc.wilson(ks[b], Mp)
                wid[b] = (hi - lo) / p
                cov[b] = lo / p <= full_rho <= hi / p
            rows.append(dict(config=name, p=p, M_sub=Mp, B=B,
                             mean_rho=ks.mean() / (Mp * p),
                             sd_rho=ks.std(ddof=1) / (Mp * p),
                             mean_ci_width=wid.mean(),
                             frac_ci_covers_full=cov.mean(),
                             full_rho=full_rho))
    print(f"{name} done", flush=True)

with open(f"{OUT}/scale_curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

INK, SURFACE = "#33322e", "#fcfcfb"
PAL = ["#8a4f9e", "#2a78d6", "#eb6834", "#2a9d64", "#c23b52", "#b8860b",
       "#5f7186", "#d06aa8"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
for ax, p in zip(axs, [1e-3, 1e-4]):
    for i, (name, _) in enumerate(sc.CONFIGS):
        rr = [r for r in rows if r["config"] == name and r["p"] == p]
        if not rr:
            continue
        ns = [r["M_sub"] for r in rr]
        mu = [r["mean_rho"] for r in rr]
        sd = [r["sd_rho"] for r in rr]
        ax.plot(ns, mu, "-o", ms=3, color=PAL[i], lw=1.3, label=name)
        ax.fill_between(ns, np.array(mu) - np.array(sd),
                        np.array(mu) + np.array(sd), color=PAL[i], alpha=0.1,
                        lw=0)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("audit sample budget M")
    ax.set_title(f"depth p = {p:g}", fontsize=10, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel(r"$\hat\rho$ (mean $\pm$ SD over subsamples)")
axs[0].legend(frameon=False, fontsize=7)
fig.suptitle("Interval width by sampling budget", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/figures/scale_curves.png", dpi=170, facecolor=SURFACE,
            bbox_inches="tight")
print("wrote scale_curves.csv + figure", flush=True)
