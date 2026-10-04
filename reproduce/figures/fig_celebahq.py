"""CelebA-HQ figure: raw and wrapped tail ratios at the three certified depths."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TAILAUDIT_DATA", os.path.join(HERE, "..", "..", "results"))
FIG_DIR = os.environ.get("TAILAUDIT_FIGS", os.path.join(HERE, "..", "figures"))
INK, SURFACE = "#33322e", "#ffffff"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a887f"
COL = {"SG2-ADA": BLUE, "DDPM-DDIM100": ORANGE}
DEPTHS = [1e-2, 3e-3, 1e-3]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


lead = [r for r in read_csv(f"{DATA_DIR}/celebahq_leaderboard.csv")
        if r["extrapolation"] == ""]
wrap = read_csv(f"{DATA_DIR}/celebahq_wrapped.csv")
models = [m for m in COL if any(r["model"] == m for r in lead)]

fig, ax = plt.subplots(figsize=(3.5, 2.7), facecolor=SURFACE)
nm = max(len(models), 1)
gw = 0.72                       # group width per depth
for gi, p in enumerate(DEPTHS):
    for mi, m in enumerate(models):
        base = gi + (mi - (nm - 1) / 2) * (gw / nm)
        r = next((r for r in lead if r["model"] == m and float(r["p"]) == p),
                 None)
        w = next((r for r in wrap if r["model"] == m and float(r["p"]) == p),
                 None)
        if r is None:
            continue
        rho = float(r["rho"])
        xr, xw = base - 0.07, base + 0.07
        # quadrature CI: thin full-range whisker
        ax.plot([xr, xr], [float(r["quad_lo"]), float(r["quad_hi"])],
                color=COL[m], lw=0.8, alpha=0.45, solid_capstyle="butt")
        # Wilson CI: heavier bar with caps
        ax.errorbar([xr], [rho],
                    yerr=[[rho - float(r["wilson_lo"])],
                          [float(r["wilson_hi"]) - rho]],
                    fmt="o", color=COL[m], ms=3.6, lw=1.3, capsize=2.2,
                    zorder=4)
        if w is not None:
            e = float(w["self_error_e"])
            ax.plot([xw - 0.055, xw + 0.055], [e, e], color=MUTED, lw=1.1,
                    ls=(0, (2, 1.2)), zorder=3)
            ax.plot([xw], [float(w["wrapped_rho"])], marker="s", ms=3.6,
                    mfc="none", mec=COL[m], mew=1.2, ls="none", zorder=4)

ax.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=1)
ax.set_xticks(range(len(DEPTHS)))
ax.set_xticklabels([r"$p=10^{-2}$", r"$p=3{\times}10^{-3}$", r"$p=10^{-3}$"],
                   fontsize=8)
ax.set_ylabel(r"tail-calibration ratio $\hat\rho(p)$", fontsize=8)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlim(-0.55, len(DEPTHS) - 0.45)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

h = [plt.Line2D([], [], marker="o", color=COL[m], ls="none", ms=3.6,
                label=m) for m in models]
h += [plt.Line2D([], [], marker="s", mfc="none", mec=INK, ls="none", ms=3.6,
                 label="wrapped"),
      plt.Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2, 1.2)),
                 label=r"split self-error $e(p)$")]
ax.legend(handles=h, frameon=False, fontsize=6.5, loc="lower left",
          ncol=2, handletextpad=0.4, columnspacing=0.9)
ax.set_title("CelebA-HQ-256: raw and wrapped tail-calibration ratio",
             fontsize=8.5, loc="left")
fig.tight_layout(pad=0.4)
fig.savefig(f"{FIG_DIR}/fig_d2.pdf", facecolor=SURFACE,
            bbox_inches="tight")
print(f"wrote {FIG_DIR}/fig_d2.pdf  (models: {models})", flush=True)
