"""Quickstart: a Frechet score cannot tell these samplers apart; the tail audit can.

Reference data: a heavy-tailed 16-d distribution (multivariate Student-t).
Three "samplers" match its mean and covariance exactly, so any score built from
first and second moments (FID-style) rates them identically:

  * faithful  - the same distribution as the data
  * too light - a lighter-tailed Student-t (too few extreme samples)
  * too heavy - a heavier Student-t (produces too many)

We audit each one, wrap the two miscalibrated ones, and audit again.

    python examples/01_quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rarecal as rc                                   # noqa: E402
from _style import plt, save, NAVY, TEAL, VIOLET, CORAL, GREY   # noqa: E402

rng = np.random.default_rng(0)
d, n = 16, 200_000


def student_t(n, df):
    z = rng.standard_normal((n, d))
    g = rng.chisquare(df, size=(n, 1)) / df
    return z / np.sqrt(g) * np.sqrt((df - 2) / df)       # unit covariance


data = student_t(3 * n, df=8)
calib, heldout = data[:n], data[n:]
samplers = {"faithful": student_t(n, 8),
            "too light": student_t(n, 14),
            "too heavy": student_t(n, 5.5)}

# 1. fit the severity functional and calibration law on reference data only
sev = rc.WhitenedRadius(n_components=d).fit(calib)
law = rc.CalibrationLaw.fit(sev(calib))
print(law, "\n")

depths = [1e-2, 1e-3]
floor = rc.frechet_distance_features(heldout[:n], heldout[n:])
print(f"real-vs-real Frechet distance (noise floor): {floor:.4f}\n")
rows = {}
for name, X in samplers.items():
    fd = rc.frechet_distance_features(heldout[:n], X)
    rep = rc.audit(sev(X), sev(heldout), depths=depths)
    Xw, info = rc.Wrapper(sev, law, tail_only=0.99)(X, return_info=True)
    rep_w = rc.audit(sev(Xw), sev(heldout), depths=depths)
    fd_w = rc.frechet_distance_features(heldout[:n], Xw)
    rows[name] = (fd, rep, fd_w, rep_w)
    print(f"=== {name}:  Frechet distance {fd:.4f}")
    print(rep)
    print(f"--- after wrapping ({info['fraction_moved']:.0%} of samples moved): "
          f"Frechet distance {fd_w:.4f}")
    print(rep_w, "\n")

# 2. figure
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1, 1.6]))
names = list(rows)
cols = [TEAL, CORAL, VIOLET]
ax = axs[0]
ax.bar(names, [rows[k][0] for k in names], color=cols, width=0.62)
ax.axhline(floor, color=GREY, lw=1.2, ls=":")
ax.text(-0.45, floor * 1.04, "real vs. real noise", va="bottom", ha="left", fontsize=8.5, color=GREY)
ax.set_ylim(0, 1.9 * max(max(rows[k][0] for k in names), floor))
for i, k in enumerate(names):
    ax.text(i, rows[k][0] * 0.5, f"{rows[k][0]:.4f}", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
ax.set_title("FID-style score: all look the same")
ax.grid(axis="x", visible=False)

ax = axs[1]
groups = [(p_i, stage) for p_i in range(len(depths)) for stage in (0, 1)]
for j, (k, c) in enumerate(zip(names, cols)):
    for stage, mk in [(0, "o"), (1, "D")]:
        rep = rows[k][1] if stage == 0 else rows[k][3]
        for p_i, t in enumerate(rep):
            xpos = p_i * 3 + stage * 1.2 + (j - 1) * 0.28
            ax.errorbar(xpos, t.rho, yerr=[[t.rho - t.lo], [t.hi - t.rho]], fmt=mk, color=c,
                        ms=7 if stage == 0 else 6, mfc=c if stage == 0 else "white", mew=1.6,
                        lw=1.6, capsize=0, label=k if (stage == 0 and p_i == 0) else None)
ax.axhline(1, color=NAVY, lw=1, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.05, 0.1, 0.3, 1, 3], ["0.05", "0.1", "0.3", "1", "3"])
ax.set_xticks([0, 1.2, 3, 4.2], ["raw", "wrapped", "raw", "wrapped"])
for p_i, pp in enumerate(depths):
    ax.text(p_i * 3 + 0.6, 7.5, f"depth p = {pp:g}", ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=NAVY)
    if p_i:
        ax.axvline(p_i * 3 - 0.9, color="#E3E5EC", lw=1)
ax.set_ylim(0.03, 9)
ax.set_ylabel(r"tail ratio $\hat\rho(p)$  (1 = right rate)")
ax.set_title("Tail audit: miscalibration is clear, and the wrapper fixes it")
ax.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.16))
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "quickstart.png")
