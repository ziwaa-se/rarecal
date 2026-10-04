"""The recalibration wrapper.

Black-box post-processing that makes a sampler's severity law equal to the
calibration law F_hat, without weights, gradients or retraining:

  1. rank the M draws by severity:            v_i = rank(r_i) / (M + 1)
  2. map each rank to a target severity:      r'_i = F_hat^{-1}(v_i)
  3. move each sample to its target severity with a transport phi_{r -> r'}
     that keeps the sample's "shape" coordinates (e.g. its direction).

Only draws with rank above `tail_only` need to move to fix the audited rates,
which keeps the change to bulk statistics such as FID of order (1 - tail_only).
"""
from __future__ import annotations

import numpy as np

from .calibration import CalibrationLaw

__all__ = ["rank_targets", "Wrapper"]


def rank_targets(r_model, law: CalibrationLaw):
    """Rank remap: returns (v, r_target) for severities r_model."""
    r_model = np.asarray(r_model, dtype=np.float64)
    M = len(r_model)
    v = (np.argsort(np.argsort(r_model)) + 1.0) / (M + 1.0)
    return v, law.ppf(v)


class Wrapper:
    """Wrap a batch of samples so their severity law matches `law`.

    Parameters
    ----------
    severity : a fitted severity functional exposing `__call__` and
        `transport(X, r_target, box=...)` (e.g. rarecal.WhitenedRadius).
    law : the calibration law fitted on reference severities.
    tail_only : move only draws whose rank exceeds this level (None = all).
    box : optional (low, high) bounds of the sample space, e.g. (0, 255).
    """

    def __init__(self, severity, law: CalibrationLaw, tail_only: float | None = 0.99,
                 box: tuple[float, float] | None = None):
        self.severity, self.law, self.tail_only, self.box = severity, law, tail_only, box

    def __call__(self, X, return_info: bool = False):
        X = np.asarray(X)
        r = self.severity(X)
        v, r_target = rank_targets(r, self.law)
        if self.tail_only is None:
            idx = np.arange(len(X))
        else:
            # every draw ranked above the level, and every draw whose raw severity
            # already exceeds the target at that level, so no exceedance is missed
            idx = np.where((v > self.tail_only) | (r > self.law.ppf(np.array([self.tail_only]))[0]))[0]
        Xw = np.array(X, dtype=np.float64, copy=True)
        if len(idx):
            Xw[idx] = self.severity.transport(Xw[idx], r_target[idx], box=self.box)
        if not return_info:
            return Xw
        r_new = self.severity(Xw[idx]) if len(idx) else np.array([])
        info = dict(moved=len(idx), fraction_moved=len(idx) / len(X),
                    median_residual=float(np.median(np.abs(r_new - r_target[idx]))) if len(idx) else 0.0,
                    r_target=r_target)
        return Xw, info
