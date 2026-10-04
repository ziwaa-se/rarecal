"""Confidence intervals for rare-event rates.

All functions return intervals for a binomial proportion q = k / M. Divide by
the tail probability p to put them on the ratio scale rho = q / p.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

__all__ = ["wald", "wilson", "clopper_pearson", "threshold_sigma", "quadrature"]


def wald(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Textbook normal-approximation interval q_hat +/- z * sqrt(q_hat (1 - q_hat) / n).

    Shown for comparison only. With few exceedances it is symmetric, can go
    below zero and has zero width at k = 0.
    """
    q = k / n
    h = z * np.sqrt(q * (1 - q) / n)
    return q - h, q + h


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    q = k / n
    d = 1 + z * z / n
    c = (q + z * z / (2 * n)) / d
    h = z * np.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (conservative) binomial interval, used as a check when k <= 1."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def threshold_sigma(p: float, n_ref: int) -> float:
    """Relative standard error of rho_hat caused by estimating tau_p from n_ref
    reference points (first order, locally constant ratio)."""
    return float(np.sqrt((1 - p) / (n_ref * p)))


def quadrature(rho: float, lo_w: float, hi_w: float, p: float, n_ref: int,
               z: float = 1.96) -> tuple[float, float]:
    """Combine the Wilson count interval [lo_w, hi_w] (rho scale) with the
    threshold-estimation error, widening each half-width in quadrature."""
    add = z * rho * threshold_sigma(p, n_ref)
    lo = rho - np.sqrt((rho - lo_w) ** 2 + add ** 2)
    hi = rho + np.sqrt((hi_w - rho) ** 2 + add ** 2)
    return float(max(lo, 0.0)), float(hi)
