"""How many samples does a rare-event audit need?

Left: 95% interval width for the tail ratio as the number of model samples M
grows, at three depths p. The dotted line marks the common 50k-sample budget.
Right: actual coverage of the textbook (Wald) interval versus the Wilson
interval for a perfectly calibrated sampler at p = 1e-4. Wald collapses when
the expected count M*p is small; Wilson stays close to 95%.

    python examples/03_sample_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rarecal as rc                                   # noqa: E402
from _style import plt, save, NAVY, TEAL, VIOLET, CORAL, AMBER, GREY   # noqa: E402

rng = np.random.default_rng(0)
Ms = np.unique(np.logspace(3.5, 6.5, 25).astype(int))
depths = [(1e-2, TEAL), (1e-3, VIOLET), (1e-4, CORAL)]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
ax = axs[0]
for p, col in depths:
    widths = []
    for M in Ms:
        k = rng.binomial(M, p, size=2000)
        w = [(rc.wilson(kk, M)[1] - rc.wilson(kk, M)[0]) / p for kk in k[:300]]
        widths.append(np.mean(w))
    ax.plot(Ms, widths, color=col, lw=2.2, label=f"p = {p:g}")
    print(f"p={p:g}: mean 95% width at M=50k: {np.interp(5e4, Ms, widths):.2f} rho-units")
ax.axvline(5e4, color=GREY, ls=":", lw=1.3)
ax.text(5.4e4, 0.03, "50k samples", color=GREY, fontsize=8.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("model samples M"); ax.set_ylabel("95% interval width (rho units)")
ax.set_title("Deeper tails need many more samples")
ax.legend(loc="upper right")

ax = axs[1]
p = 1e-4
cov_wald, cov_wil, Mp = [], [], []
for M in Ms:
    k = rng.binomial(M, p, size=4000)
    wa = np.array([rc.wald(kk, M) for kk in k]); wi = np.array([rc.wilson(kk, M) for kk in k])
    cov_wald.append(np.mean((wa[:, 0] <= p) & (p <= wa[:, 1])))
    cov_wil.append(np.mean((wi[:, 0] <= p) & (p <= wi[:, 1])))
    Mp.append(M * p)
ax.plot(Mp, cov_wald, color=AMBER, lw=2.2, label="Wald (textbook)")
ax.plot(Mp, cov_wil, color=VIOLET, lw=2.2, label="Wilson (used here)")
ax.axhline(0.95, color=NAVY, ls="--", lw=1)
ax.set_xscale("log"); ax.set_ylim(0.5, 1.0)
ax.set_xlabel(r"expected exceedances $M\cdot p$"); ax.set_ylabel("actual coverage of a 95% interval")
ax.set_title("Few exceedances: the textbook interval fails")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "sample_budget.png")

for c in (2, 3):
    print(f"any test needs n >= {rc.sample_size_lower_bound(1e-4, c):,.0f} draws "
          f"to tell rho = 1 from rho = {c} at p = 1e-4 (lower bound, constant 1/18)")
