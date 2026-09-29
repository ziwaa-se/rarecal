"""The blind spot, constructed exactly.

A Frechet score (FID, FMD, FD-DINOv2, ...) sees a sampler only through the mean
and covariance of its features. We build two samplers whose features have
IDENTICAL mean and covariance, yet one produces c times more extreme samples
per unit of tail probability than the other.

Recipe (Proposition 1 in the accompanying paper):
  Q_T : reference samples beyond the tail threshold tau_p
  Q_B : a weighted set of bulk samples (below tau_p) whose feature moments
        equal those of Q_T  (found by non-negative least squares)
  Q_1 = (1 - eps) Q_0 + eps Q_B      Q_2 = (1 - eps) Q_0 + eps Q_T,   eps = c * p

Here the severity uses all 3 coordinates, while the "feature extractor" of
the metric sees only the first two, as when an embedding network discards
some of the information that makes a sample extreme.

    python examples/02_blind_spot.py
"""
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rarecal as rc                                   # noqa: E402
from _style import plt, save, NAVY, TEAL, VIOLET, CORAL, GREY   # noqa: E402

rng = np.random.default_rng(1)
n, p, c = 200_000, 1e-2, 3.0
ref = rng.standard_normal((n, 3)) * [1.0, 1.0, 1.0]
ref[:, 2] = rng.standard_t(5, n) / np.sqrt(5 / 3)          # heavier third coordinate
features = lambda X: X[:, :2]                               # what the metric sees


def psi(X):                                                 # FID's moment map
    f = features(X)
    return np.column_stack([f, f[:, 0] ** 2, f[:, 0] * f[:, 1], f[:, 1] ** 2])


sev = rc.WhitenedRadius(3).fit(ref)
r = sev(ref)
tau = np.quantile(r, 1 - p)
tail, bulk = ref[r >= tau], ref[r < tau]

# Q_B: non-negative weights on 4,000 bulk candidates matching the tail's moments
cand = bulk[rng.choice(len(bulk), 4000, replace=False)]
A = np.vstack([psi(cand).T, np.ones(len(cand))])
b = np.append(psi(tail).mean(0), 1.0)
w, resid = nnls(A, b)
w /= w.sum()
print(f"tail points: {len(tail)};  bulk atoms used by Q_B: {(w > 1e-9).sum()};  "
      f"moment residual {resid:.2e}")


def moments(parts):
    """Exact feature mean/cov of a mixture given (weight, points, point_weights)."""
    mu = sum(a * (pw[:, None] * features(P)).sum(0) for a, P, pw in parts)
    m2 = sum(a * np.einsum("i,ij,ik->jk", pw, features(P), features(P)) for a, P, pw in parts)
    return mu, m2 - np.outer(mu, mu)


def tail_mass(parts):
    return sum(a * (pw * (sev(P) >= tau)).sum() for a, P, pw in parts)


eps = c * p
base = ref[rng.choice(n, 50_000, replace=False)]
u_base = np.full(len(base), 1 / len(base))
Q1 = [(1 - eps, base, u_base), (eps, cand, w)]
Q2 = [(1 - eps, base, u_base), (eps, tail, np.full(len(tail), 1 / len(tail)))]
mu_P, S_P = features(ref).mean(0), np.cov(features(ref), rowvar=False, bias=True)
out = {}
for name, Q in [("Q1 (mix in bulk)", Q1), ("Q2 (mix in tail)", Q2)]:
    mu, S = moments(Q)
    out[name] = (rc.frechet_distance(mu_P, S_P, mu, S), tail_mass(Q) / p)
    print(f"{name}: Frechet distance {out[name][0]:.6f}   tail ratio rho(p) = {out[name][1]:.3f}")
print(f"difference in rho: {out['Q2 (mix in tail)'][1] - out['Q1 (mix in bulk)'][1]:.3f}  (theory: c = {c})")

# figure: the two mixtures share a feature ellipse but not a tail
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4))
ax = axs[0]
ax.scatter(*features(base[:3000]).T, s=3, color="#D5D8E3", label="base sampler")
ax.scatter(*features(tail[:700]).T, s=7, color=CORAL, alpha=0.55, lw=0, label=r"$Q_T$: tail samples")
k = w > 1e-9
ax.scatter(*features(cand[k]).T, s=40 + 700 * w[k], color=TEAL, edgecolor=NAVY, lw=0.8, zorder=5,
           label=r"$Q_B$: weighted bulk atoms")
mT, ST = moments([(1, tail, np.full(len(tail), 1 / len(tail)))])
th = np.linspace(0, 2 * np.pi, 200)
L = np.linalg.cholesky(ST)
ell = mT[:, None] + 2 * L @ np.vstack([np.cos(th), np.sin(th)])
ax.plot(*ell, color=NAVY, lw=1.6, label="shared mean & covariance")
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_title("What the metric sees: identical moments")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, markerscale=1.0)
ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)

ax = axs[1]
names = list(out)
ax.bar([0, 1], [out[k][1] for k in names], color=[TEAL, CORAL], width=0.55)
for i, k in enumerate(names):
    ax.text(i, out[k][1], f"rho = {out[k][1]:.2f}\nFD = {out[k][0]:.4f}", ha="center", va="bottom", fontsize=9)
ax.axhline(1, color=NAVY, ls="--", lw=1)
ax.set_xticks([0, 1], ["mix in bulk", "mix in tail"])
ax.set_ylabel(r"tail ratio $\rho(p)$")
ax.set_ylim(0, 1.35 * max(out[k][1] for k in names))
ax.set_title(f"Same Frechet distance, tail ratios differ by c = {c:g}")
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "blind_spot.png")
