"""Figure 1: audited tail ratio against measured FMD and against published FID."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TAILAUDIT_DATA", os.path.join(HERE, "..", "..", "results"))
FIGS = os.environ.get("TAILAUDIT_FIGS", os.path.join(HERE, "..", "figures"))
INK, MUTED, SURFACE = "#33322e", "#8a887f", "#ffffff"
PAL8 = {"iDDPM": "#8a4f9e", "EDM": "#2a78d6", "EDM-churn": "#eb6834",
        "EDM2-S": "#2a9d64", "EDM2-M": "#8f2358", "StyleGAN-XL": "#b8860b",
        "DiT-cfg1.5": "#3a6db8", "DiT-cfg1.0": "#d06aa8"}
MRK8 = {"iDDPM": "o", "EDM": "s", "EDM-churn": "^", "EDM2-S": "D",
        "EDM2-M": "v", "StyleGAN-XL": "P", "DiT-cfg1.5": "X", "DiT-cfg1.0": "*"}
plt.rcParams.update({
    "font.size": 11.0, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 11.5, "axes.labelsize": 10.5, "legend.fontsize": 9.0,
    "pdf.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE})

F = float
rows = list(csv.DictReader(open(f"{DATA}/quality_axis_fmd.csv")))
# EDM-churn and EDM sit almost on top of each other on the FMD axis; push the
# churn label up-left and EDM right so the two do not overprint.
OFF_L = {"iDDPM": (-34, 8), "EDM": (8, 6), "EDM-churn": (-30, 12),
         "EDM2-S": (6, 2), "EDM2-M": (6, -12), "StyleGAN-XL": (-52, -2),
         "DiT-cfg1.5": (7, 5), "DiT-cfg1.0": (-40, -4)}
OFF_R = {"iDDPM": (6, 4), "EDM": (6, 4), "EDM2-S": (6, 4),
         "EDM2-M": (6, -12), "StyleGAN-XL": (5, 6), "DiT-cfg1.5": (6, 5),
         "DiT-cfg1.0": (-40, 8)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.3))


def draw(ax, xs, rs, cfgs, los, his, off):
    for x, r, cfg, lo, hi in zip(xs, rs, cfgs, los, his):
        ax.errorbar(x, r, yerr=[[r - lo], [hi - r]], fmt=MRK8[cfg], ms=6,
                    color=PAL8[cfg], capsize=2.5, lw=1.2, mew=0.8, zorder=3)
        ax.annotate(cfg, (x, r), textcoords="offset points",
                    xytext=off[cfg], fontsize=8.6, color=INK)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=1)
    ax.text(0.985, 1.0, "calibrated", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.6, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(xs) * 0.62, max(xs) * 1.9)
    ax.set_ylim(0.085, 4.0)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax.set_ylabel(r"tail-calibration ratio  $\hat\rho_{R_1}(10^{-3})$")
    ax.grid(alpha=0.25, lw=0.5)


# -------- left: FMD measured here -------------------------------------------
xs = [F(r["fmd_mae_auditedM"]) for r in rows]
rs = [F(r["rho_s1_1e3"]) for r in rows]
lo = [F(r["rho_quad_lo"]) for r in rows]
hi = [F(r["rho_quad_hi"]) for r in rows]
cf = [r["config"] for r in rows]
draw(ax1, xs, rs, cf, lo, hi, OFF_L)
fl = [F(r["fmd_real_floor_sameM"]) for r in rows]
sp1, kt1 = spearmanr(xs, rs), kendalltau(xs, rs)
al1 = spearmanr(xs, [F(r["abs_log_rho"]) for r in rows])
ax1.set_xlabel("FMD (audited $32^2$ sets)  " r"$\longrightarrow$ worse")
ax1.set_title("Commensurable axis (measured here)\n"
              rf"$n=8$: ${sp1.statistic:+.2f}$ vs $\rho$; "
              rf"${al1.statistic:+.2f}$ vs $|\log\rho|$",
              loc="left", fontweight="bold")
ax1.text(0.015, 0.03,
         f"real-vs-real FMD floor at the same $M$: {min(fl):.4f}–{max(fl):.3f}"
         f"\n(≥{min(xs)/max(fl):.0f}× below the smallest model FMD)",
         transform=ax1.transAxes, fontsize=7.6, color=MUTED, va="bottom")

# -------- right: published FID ----------------------------------------------
fr = [r for r in rows if r["fid_published"]]
xs2 = [F(r["fid_published"]) for r in fr]
rs2 = [F(r["rho_s1_1e3"]) for r in fr]
draw(ax2, xs2, rs2, [r["config"] for r in fr],
     [F(r["rho_quad_lo"]) for r in fr], [F(r["rho_quad_hi"]) for r in fr],
     OFF_R)
sp2, kt2 = spearmanr(xs2, rs2), kendalltau(xs2, rs2)
al2 = spearmanr(xs2, [F(r["abs_log_rho"]) for r in fr])
ax2.set_xlabel("published FID (own resolution)  "
               r"$\longrightarrow$ worse")
ax2.set_title("Published FID (not commensurable)\n"
              rf"$n=7$: ${sp2.statistic:+.2f}$ vs $\rho$; "
              rf"${al2.statistic:+.2f}$ vs $|\log\rho|$",
              loc="left", fontweight="bold")
res = sorted({r["fid_resolution"] for r in fr if r["fid_resolution"]})
ax2.text(0.015, 0.03, "resolutions mixed: " + ", ".join(res) +
         "\nEDM-churn omitted (no published FID)",
         transform=ax2.transAxes, fontsize=7.6, color=MUTED, va="bottom")

fig.tight_layout(w_pad=2.0)
out = f"{FIGS}/fig_quality_axis.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
cross = spearmanr(xs2, [F(r["fmd_mae_auditedM"]) for r in fr])
print(f"  FMD vs rho      : spearman {sp1.statistic:+.4f} (p={sp1.pvalue:.4f})"
      f"  kendall {kt1.statistic:+.4f} (p={kt1.pvalue:.4f})  n=8")
print(f"  FMD vs |log rho|: spearman {al1.statistic:+.4f} "
      f"(p={al1.pvalue:.4f})  n=8")
print(f"  pubFID vs rho   : spearman {sp2.statistic:+.4f} (p={sp2.pvalue:.4f})"
      f"  kendall {kt2.statistic:+.4f} (p={kt2.pvalue:.4f})  n=7")
print(f"  pubFID vs |log rho|: spearman {al2.statistic:+.4f} "
      f"(p={al2.pvalue:.4f})  n=7")
print(f"  pubFID vs FMD (do the two quality axes agree?): "
      f"spearman {cross.statistic:+.4f} (p={cross.pvalue:.4f})  n=7")
