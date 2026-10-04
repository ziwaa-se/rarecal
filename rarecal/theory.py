"""Small numerical companions to the theory.

* `frechet_distance`: the Frechet (2-Wasserstein between Gaussian fits) distance
  that FID and FMD are built on.
* `sample_size_lower_bound`: draws any test needs to tell a tail rate p from c*p.
* `frechet_cost_bound`: how much any post-processing can move a Frechet score.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

__all__ = ["frechet_distance", "frechet_distance_features", "sample_size_lower_bound",
           "frechet_cost_bound"]


def frechet_distance(mu1, S1, mu2, S2) -> float:
    """Squared Frechet distance ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S1^1/2 S2 S1^1/2)^1/2)."""
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    r1 = sqrtm(S1)
    cross = sqrtm(r1 @ S2 @ r1)
    val = float(np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.trace(np.real(cross)))
    return max(val, 0.0)


def frechet_distance_features(F1, F2) -> float:
    """FID-style score between two feature matrices (n, d)."""
    F1, F2 = np.asarray(F1, np.float64), np.asarray(F2, np.float64)
    return frechet_distance(F1.mean(0), np.cov(F1, rowvar=False), F2.mean(0), np.cov(F2, rowvar=False))


def sample_size_lower_bound(p: float, c: float, C: float = 1 / 18) -> float:
    """n >= C / (p (sqrt(c) - 1)^2): draws needed by ANY test to distinguish a
    tail rate p from c*p (two-point Le Cam bound; requires c*p <= 1/2)."""
    return C / (p * (np.sqrt(c) - 1) ** 2)


def frechet_cost_bound(feat_before, feat_after) -> float:
    """Upper bound on |d_F(P, Q_wrapped) - d_F(P, Q)| where d_F = sqrt(FID):
    the root-mean-square feature displacement of the post-processing. If only
    a fraction pi of samples move, this equals sqrt(pi) * RMS over moved samples."""
    d = np.asarray(feat_after, np.float64) - np.asarray(feat_before, np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
