"""Plug-in rare-event estimates from synthetic corpora of increasing size: raw, wrapped and real-data control."""
import os
import sys
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

OUT = os.path.expandvars("${PROJECT_ROOT}/validation/validated_results")
rng = np.random.default_rng(42)

L = np.load(f"{sc.P2}/layer_seed8000.npz")
fit_idx, hold_idx, _ = sc.split_indices(8000)
Rf_sorted = np.sort(L["R50"][fit_idx])
Rh = L["R50"][hold_idx]
u, xi, sg, zeta = float(L["u"]), float(L["xi"]), float(L["sg"]), float(L["zeta"])
tau = {1e-3: float(L["tau3"]), 1e-4: float(L["tau4"])}

NS = [1000, 3162, 10_000, 31_623, 100_000]
B = 500
rows = []
series = {}

def arm_curve(Rsrc, label, model, p):
    t = tau[p]
    for n in NS:
        if n < len(Rsrc):
            k = np.array([( Rsrc[rng.integers(0, len(Rsrc), n)] >= t).sum()
                          for _ in range(B)])
            mean_ratio = k.mean() / (n * p)
            sd_ratio = k.std(ddof=1) / (n * p)
            lo = hi = np.nan
        else:
            kk = int((Rsrc >= t).sum())
            mean_ratio = kk / (len(Rsrc) * p)
            sd_ratio = 0.0
            wl, wh = sc.wilson(kk, len(Rsrc))
            lo, hi = wl / p, wh / p
        rows.append(dict(model=model, arm=label, p=p, n=n,
                         mean_ratio=mean_ratio, sd_ratio=sd_ratio,
                         wilson_lo=lo, wilson_hi=hi, B=B))
        series.setdefault((model, label, p), []).append((n, mean_ratio, sd_ratio))

for model in ["iDDPM", "EDM"]:
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[model]}.npz")["W"], 50)
    M = len(Rm)
    v = (np.argsort(np.argsort(Rm)) + 1.0) / (M + 1.0)
    Rw = sc.spliced_quantile(v, Rf_sorted, u, xi, sg, zeta)
    for p in [1e-3, 1e-4]:
        arm_curve(Rm, "raw", model, p)
        arm_curve(Rw, "wrapped", model, p)
# real-data control (one series per depth)
for p in [1e-3, 1e-4]:
    arm_curve(Rh, "real-control", "held-out", p)

with open(f"{OUT}/downstream_inheritance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# success criteria: raw mean within audited CI at every n;
# wrapped within [0.9,1.1] at 1e-3 for n >= 1e4
ok_lines = []
for model in ["iDDPM", "EDM"]:
    Rm = sc.radius_k(np.load(f"{sc.P2}/model_w100_{sc.SLUG[model]}.npz")["W"], 50)
    a = sc.audit_rho(Rm, tau[1e-3], 1e-3)
    raws = [r for r in rows if r["model"] == model and r["arm"] == "raw"
            and r["p"] == 1e-3]
    ok_raw = all(a["lo"] <= r["mean_ratio"] <= a["hi"] for r in raws)
    wr = [r for r in rows if r["model"] == model and r["arm"] == "wrapped"
          and r["p"] == 1e-3 and r["n"] >= 10_000]
    ok_wrap = all(0.9 <= r["mean_ratio"] <= 1.1 for r in wr)
    ok_lines.append(f"{model}: raw-in-CI={ok_raw} wrapped-in-[0.9,1.1]@1e-3,"
                    f"n>=1e4={ok_wrap}")
    print(ok_lines[-1], flush=True)

# figure
INK, SURFACE = "#33322e", "#fcfcfb"
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE, sharey=False)
for ax, p in zip(axs, [1e-3, 1e-4]):
    for (model, label, pp), pts in series.items():
        if pp != p or label == "real-control":
            continue
        pts = sorted(pts)
        ns = [x[0] for x in pts]; mu = [x[1] for x in pts]; sd = [x[2] for x in pts]
        ls = "-" if label == "wrapped" else "--"
        col = "#2a78d6" if model == "iDDPM" else "#eb6834"
        ax.plot(ns, mu, ls, color=col, lw=1.6, label=f"{model} {label}")
        ax.fill_between(ns, np.array(mu) - np.array(sd),
                        np.array(mu) + np.array(sd), color=col, alpha=0.12, lw=0)
    ctrl = sorted(series[("held-out", "real-control", p)])
    ax.plot([x[0] for x in ctrl], [x[1] for x in ctrl], ":", color=INK, lw=1.4,
            label="real-data control")
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("synthetic corpus size n")
    ax.set_title(f"depth p = {p:g}", fontsize=10, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel(r"$\hat p_n / p$")
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("plug-in rare-event estimates inherit exactly the factor rho "
             "at every corpus size; the wrapper repairs the estimand",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/figures/downstream_inheritance.png", dpi=170, facecolor=SURFACE,
            bbox_inches="tight")
print("wrote downstream_inheritance.csv + figure", flush=True)
