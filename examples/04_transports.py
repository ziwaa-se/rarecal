"""Moving a sample to a target severity without retraining anything.

Left: the whitened radius. Its transport rescales the whitened coordinates, so
samples slide along straight rays from the centre (in whitened and data space)
and keep their direction, which is the "shape" the wrapper leaves untouched.

Right: a curved severity R(x) = x1 + x2^2 / 4 with no closed-form transport.
The normalized gradient flow dx/ds = grad R / ||grad R||^2 raises R at unit
speed, so integrating it for time r' - r lands exactly on the target level.

    python examples/04_transports.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rarecal as rc                                   # noqa: E402
from _style import plt, save, NAVY, TEAL, VIOLET, CORAL, GREY   # noqa: E402

rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))

# ---- whitened radius: closed-form radial transport ---------------------------
ax = axs[0]
A = np.array([[1.6, 0.0], [0.9, 0.6]])
X = rng.standard_normal((20_000, 2)) @ A.T
sev = rc.WhitenedRadius(2).fit(X)
g = np.linspace(-6, 6, 300)
GX, GY = np.meshgrid(g, g)
Z = sev(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
ax.scatter(*X[:2500].T, s=2, color="#D5D8E3")
cs = ax.contour(GX, GY, Z, levels=[1, 2, 3, 4], colors=[GREY], linewidths=0.8)
ax.clabel(cs, fmt="R=%g", fontsize=7)
starts = X[rng.choice(len(X), 60, replace=False)]
starts = starts[(sev(starts) > 0.5) & (sev(starts) < 1.8)][:7]
for x0 in starts:
    path = np.array([sev.transport(x0[None], np.array([t]))[0] for t in np.linspace(sev(x0[None])[0], 3.6, 30)])
    ax.plot(*path.T, color=TEAL, lw=2)
    ax.scatter(*x0, color=NAVY, s=18, zorder=5)
    ax.scatter(*path[-1], color=CORAL, s=30, marker="*", zorder=5)
ax.set_title("Whitened radius: exact radial transport")
ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)

# ---- curved severity: gradient flow ------------------------------------------
ax = axs[1]
R = lambda Y: Y[:, 0] + Y[:, 1] ** 2 / 4
grad = lambda Y: np.stack([np.ones(len(Y)), Y[:, 1] / 2], 1)
g1, g2 = np.linspace(-3, 6, 300), np.linspace(-4, 4, 300)
GX, GY = np.meshgrid(g1, g2)
Z = R(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
ax.contourf(GX, GY, Z, levels=20, cmap="Purples", alpha=0.35)
cs = ax.contour(GX, GY, Z, levels=[-1, 0, 1, 2, 3], colors=[NAVY], linewidths=0.8)
ax.clabel(cs, fmt="R=%g", fontsize=7)
starts = np.column_stack([rng.uniform(-2.2, -0.8, 8), rng.uniform(-2.2, 2.2, 8)])
for x0 in starts:
    ts = np.linspace(R(x0[None])[0], 3.0, 25)
    path = np.array([rc.gradient_flow(x0, t, R, grad, steps=100) for t in ts])
    ax.plot(*path.T, color=VIOLET, lw=2)
    ax.scatter(*x0, color=NAVY, s=18, zorder=5)
    ax.scatter(*path[-1], color=CORAL, s=30, marker="*", zorder=5)
    assert abs(R(path[-1][None])[0] - 3.0) < 1e-6
ax.set_title(r"Curved severity: gradient flow to $R = 3$")
ax.set_xlim(-3, 6); ax.set_ylim(-4, 4)
fig.tight_layout()
save(fig, "transports.png")
print("all gradient-flow endpoints land on the target level (|R - 3| < 1e-6)")
