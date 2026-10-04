"""Draw the README figures for the real-model study from the CSVs in results/.

No GPU and no model weights needed; every number comes from results/*.csv.

    python scripts/make_results_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
from _style import plt, save, NAVY, TEAL, VIOLET, CORAL, AMBER, GREY   # noqa: E402

RES = os.path.join(HERE, "..", "results")
VCOL = {"under": AMBER, "calibrated": TEAL, "over": CORAL}
ORDER = ["StyleGAN-XL", "EDM2-M", "EDM2-S", "DiT-cfg1.5", "EDM", "iDDPM", "DiT-cfg1.0", "EDM-churn"]


def label(cfg):
    return {"DiT-cfg1.5": "DiT-XL/2 (cfg 1.5)", "DiT-cfg1.0": "DiT-XL/2 (no cfg)",
            "EDM-churn": "EDM (stochastic)"}.get(cfg, cfg)


# ---------------------------------------------------------------- 1. hero plot
lb = pd.read_csv(os.path.join(RES, "leaderboard_r1.csv"))
d3 = lb[lb.p == 1e-3].set_index("config").loc[ORDER]
fig, ax = plt.subplots(figsize=(8.6, 4.1))
y = np.arange(len(d3))[::-1]
ax.axvline(1, color=NAVY, ls="--", lw=1)
for yi, (cfg, r) in zip(y, d3.iterrows()):
    c = VCOL[r.verdict_quad]
    ax.plot([r.quad_lo, r.quad_hi], [yi, yi], color=c, lw=3.2, solid_capstyle="round", alpha=0.9)
    ax.scatter(r.rho, yi, s=70, color=c, edgecolor=NAVY, lw=0.8, zorder=5)
    ax.text(3.35, yi, f"{r.rho:.2f}", va="center", ha="left", fontsize=9, color=c, fontweight="bold")
    fid = "n/a" if pd.isna(r.fid_published) else f"{float(r.fid_published):.2f}"
    ax.text(0.085, yi, fid, va="center", ha="left", fontsize=9, color=GREY)
ax.set_yticks(y, [label(c) for c in d3.index])
ax.set_xscale("log")
ax.set_xlim(0.08, 4.2)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
ax.text(0.085, len(d3) - 0.35, "FID", fontsize=9, color=GREY, fontweight="bold")
ax.text(3.35, len(d3) - 0.35, r"$\hat\rho$", fontsize=10, color=NAVY, fontweight="bold")
ax.set_xlabel(r"tail ratio $\hat\rho$ at depth $p=10^{-3}$   (model rate / real rate of 1-in-1000 extremes)")
ax.set_ylim(-0.7, len(d3) - 0.1)
ax.grid(axis="y", visible=False)
for v, xx in [("under", 0.25), ("calibrated", 1.0), ("over", 2.4)]:
    ax.text(xx, -0.62, {"under": "too few extremes", "calibrated": "right rate",
                        "over": "too many extremes"}[v], ha="center", fontsize=8.5, color=VCOL[v])
ax.set_title("Eight ImageNet samplers, ordered by FID: the tail tells a different story", loc="left")
fig.tight_layout()
save(fig, "hero_leaderboard.png")

# ------------------------------------------------ 2. wrapper: fixed, FID intact
dt = pd.read_csv(os.path.join(RES, "wrap_fid", "wrapfid_delta_table.csv"))
tail = dt[dt.arm == "wrap-tail"].set_index("config")
full = dt[dt.arm == "wrap-full"].set_index("config")
cfgs = ["iDDPM", "EDM", "EDM2-S", "DiT-cfg1.0", "EDM2-M", "DiT-cfg1.5", "StyleGAN-XL"]
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
y = np.arange(len(cfgs))[::-1]
ax.axvline(1, color=NAVY, ls="--", lw=1)
for yi, c in zip(y, cfgs):
    r0, r1 = tail.loc[c, "rho3_raw"], tail.loc[c, "rho3_arm"]
    ax.annotate("", xy=(r1, yi), xytext=(r0, yi),
                arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.8, shrinkA=6, shrinkB=5))
    ax.scatter(r0, yi, s=60, color=VCOL[tail.loc[c, "verdict_raw"]], edgecolor=NAVY, lw=0.7, zorder=5)
    ax.scatter(r1, yi, s=60, marker="D", color=TEAL, edgecolor=NAVY, lw=0.7, zorder=5)
ax.set_yticks(y, [label(c) for c in cfgs])
ax.set_xscale("log"); ax.set_xlim(0.13, 2.6)
ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
ax.set_xlabel(r"tail ratio $\hat\rho(10^{-3})$:  raw $\rightarrow$ wrapped")
ax.set_title("Wrapper brings every model to the right rate", loc="left")
ax.grid(axis="y", visible=False)

ax = axs[1]
w = 0.38
ax.barh(y + w / 2, full.loc[cfgs, "dFID_pct"].abs(), height=w, color="#C9CCD6", label="move every sample")
ax.barh(y - w / 2, tail.loc[cfgs, "dFID_pct"].abs(), height=w, color=VIOLET, label="move only the top 1% (default)")
for yi, c in zip(y, cfgs):
    ax.text(abs(tail.loc[c, "dFID_pct"]) * 1.25 + 1e-3, yi - w / 2, f"{abs(tail.loc[c, 'dFID_pct']):.3f}%",
            va="center", fontsize=8, color=VIOLET)
ax.set_xscale("log"); ax.set_xlim(5e-4, 40)
ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10], ["0.001%", "0.01%", "0.1%", "1%", "10%"])
ax.set_yticks(y, [""] * len(cfgs))
ax.set_xlabel("|change in FID|")
ax.set_title("...while FID barely moves", loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.34), ncol=2, fontsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "wrapper_results.png")

# ------------------------------------------ 3. severity functionals disagree
lf = pd.read_csv(os.path.join(RES, "leaderboard_functional.csv"))
names = {"S1": "pixel radius", "S2": "MAE embedding\nradius", "S3": "max patch\nradius"}
cfgs = ORDER
fig, ax = plt.subplots(figsize=(8.2, 3.3))
for j, (s, sname) in enumerate(names.items()):
    for i, c in enumerate(cfgs):
        r = lf[(lf.config == c) & (lf.functional == s)].iloc[0]
        ax.add_patch(plt.Rectangle((i - 0.46, 2 - j - 0.44), 0.92, 0.88, color=VCOL[r.verdict],
                                   alpha=0.4, lw=0))
        ax.text(i, 2 - j, f"{r.rho:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
ax.set_xlim(-0.55, len(cfgs) - 0.45); ax.set_ylim(-0.55, 2.55)
ax.set_xticks(range(len(cfgs)), [label(c).replace(" (", "\n(") for c in cfgs], fontsize=8.5)
ax.set_yticks([2, 1, 0], list(names.values()), fontsize=9)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title(r"Same models, three definitions of 'extreme': $\hat\rho(10^{-3})$", loc="left")
for i, (v, t) in enumerate([("under", "too few extremes"), ("calibrated", "right rate"), ("over", "too many")]):
    ax.add_patch(plt.Rectangle((1.6 + 2.1 * i, -1.62), 0.25, 0.25, color=VCOL[v], alpha=0.4, lw=0, clip_on=False))
    ax.text(1.95 + 2.1 * i, -1.495, t, va="center", fontsize=9, color=NAVY)
fig.tight_layout()
save(fig, "functionals.png")

# ------------------------------------------------------- 4. tail shape collapse
sd = pd.read_csv(os.path.join(RES, "shape_dispersion.csv"))
sd = sd[(sd.seed == 20260816) & (sd.depth == 1e-3)].set_index("config")
fig, ax = plt.subplots(figsize=(7.4, 3.4))
x = np.arange(len(ORDER))
for k, (pc, col, dx) in enumerate([("PC1", VIOLET, -0.14), ("PC2", TEAL, 0.14)]):
    s = sd[sd.pc == pc].loc[ORDER]
    ax.errorbar(x + dx, s.sd_ratio, yerr=[s.sd_ratio - s.ratio_lo, s.ratio_hi - s.sd_ratio],
                fmt="o", color=col, ms=6, lw=1.8, capsize=0, label=f"along {pc}")
ax.axhline(1, color=NAVY, ls="--", lw=1)
ax.text(len(ORDER) - 0.5, 1.03, "same spread as real extremes", ha="right", va="bottom", fontsize=8.5, color=NAVY)
ax.set_xticks(x, [label(c).replace(" (", "\n(") for c in ORDER], fontsize=8.5)
ax.set_ylim(0, 1.25)
ax.set_ylabel("spread of generated extremes\n/ spread of real extremes")
ax.set_title(r"Generated extremes are less varied than real ones ($p=10^{-3}$)", loc="left")
ax.legend(loc="lower right", ncol=2)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "shape_collapse.png")

# ------------------------------------------------------------- key numbers
rej = pd.read_csv(os.path.join(RES, "shape_dispersion.csv"))
rej = rej[rej.seed == 20260816]
p_sorted = np.sort(rej.p_bf.values)                          # Brown-Forsythe, Holm step-down
holm = np.maximum.accumulate(np.minimum(1, (len(p_sorted) - np.arange(len(p_sorted))) * p_sorted))
print(f"hero: rho(1e-3) spans {d3.rho.min():.2f} .. {d3.rho.max():.2f}  "
      f"({d3.rho.max() / d3.rho.min():.1f}x); verdicts {d3.verdict_quad.value_counts().to_dict()}")
print(f"wrapper: max |dFID| tail-only {tail.dFID_pct.abs().max():.3f}%, full {full.dFID_pct.abs().max():.2f}%; "
      f"max fraction moved {tail.frac_modified.max():.2%}")
print(f"shape: {int((holm < 0.05).sum())}/{len(rej)} Brown-Forsythe tests reject equal spread "
      f"(Holm, 5%); permutation test: {int(rej.reject_05.sum())}/{len(rej)}")
