"""The calibration law: empirical bulk spliced to a generalized-Pareto tail.

F_hat(r) equals the empirical CDF of reference severities below a threshold u
and a fitted generalized Pareto distribution (GPD) above it. The threshold is
chosen automatically from a small grid of upper quantiles by a
Kolmogorov-Smirnov goodness-of-fit score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto, kstest

__all__ = ["CalibrationLaw"]

DEFAULT_GRID = (0.98, 0.985, 0.99, 0.9925, 0.995)


def _fit_gpd(excess):
    xi, _, sigma = genpareto.fit(excess, floc=0.0)
    score = kstest(excess, "genpareto", args=(xi, 0.0, sigma)).pvalue
    return xi, sigma, score


@dataclass
class CalibrationLaw:
    sorted_ref: np.ndarray
    q: float
    u: float
    xi: float
    sigma: float
    zeta: float
    ks_score: float

    @classmethod
    def fit(cls, r_ref, q_grid=DEFAULT_GRID, q: float | None = None) -> "CalibrationLaw":
        """Fit on reference severities. Pass `q` to fix the threshold quantile."""
        r = np.sort(np.asarray(r_ref, dtype=np.float64))
        best = None
        grid = [q] if q is not None else list(q_grid)
        for j, qq in enumerate(grid):
            u = float(np.quantile(r, qq))
            xi, sigma, score = _fit_gpd(r[r > u] - u)
            key = score + 1e-12 * j          # ties go to the deeper threshold
            if best is None or key > best[0]:
                best = (key, qq, u, xi, sigma, score)
        _, qq, u, xi, sigma, score = best
        zeta = float((r > u).mean())
        return cls(r, qq, u, xi, sigma, zeta, score)

    # ---- distribution functions ---------------------------------------------
    def sf(self, t) -> np.ndarray:
        """Survival function 1 - F_hat(t)."""
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        lo = t <= self.u
        out[lo] = 1.0 - np.searchsorted(self.sorted_ref, t[lo], side="right") / len(self.sorted_ref)
        out[~lo] = self.zeta * genpareto.sf(t[~lo] - self.u, self.xi, 0.0, self.sigma)
        return out

    def cdf(self, t) -> np.ndarray:
        return 1.0 - self.sf(t)

    def ppf(self, v) -> np.ndarray:
        """Quantile function F_hat^{-1}(v)."""
        v = np.asarray(v, dtype=np.float64)
        out = np.empty_like(v)
        bulk = v <= 1.0 - self.zeta
        out[bulk] = np.quantile(self.sorted_ref, np.clip(v[bulk], 0, 1))
        e = 1.0 - v[~bulk]
        if abs(self.xi) < 1e-9:
            out[~bulk] = self.u + self.sigma * np.log(self.zeta / e)
        else:
            out[~bulk] = self.u + (self.sigma / self.xi) * ((e / self.zeta) ** (-self.xi) - 1.0)
        return out

    def ratio(self, tau: float, p: float) -> float:
        """Calibration-law ratio e(p) = (1 - F_hat(tau_p)) / p: the tail ratio a
        perfectly wrapped sampler inherits at reference threshold tau_p."""
        return float(self.sf(np.array([tau]))[0] / p)

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return self.ppf(rng.uniform(size=n))

    def __repr__(self):
        return (f"CalibrationLaw(q={self.q}, u={self.u:.4g}, xi={self.xi:+.3f}, "
                f"sigma={self.sigma:.3g}, zeta={self.zeta:.4g}, KS score={self.ks_score:.3f})")
