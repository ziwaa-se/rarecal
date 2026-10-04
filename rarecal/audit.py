"""Tail-calibration audit of a black-box sampler.

Given severities of reference samples (held out) and of model samples, the
audit estimates, at each tail probability p,

    rho(p) = Q{R >= tau_p} / P{R >= tau_p},   tau_p = upper-p quantile of R under P,

with a Wilson interval for the exceedance count, widened in quadrature for
the error in estimating tau_p from the reference split.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .intervals import wilson, quadrature, clopper_pearson

__all__ = ["TailRatio", "AuditReport", "audit", "verdict"]


def verdict(lo: float, hi: float) -> str:
    """'under' / 'over' when the interval lies below / above 1, else 'calibrated'
    (calibration is not rejected; this does not certify rho = 1)."""
    if hi < 1.0:
        return "under"
    if lo > 1.0:
        return "over"
    return "calibrated"


@dataclass
class TailRatio:
    p: float
    tau: float
    k: int
    M: int
    rho: float
    lo: float
    hi: float
    lo_wilson: float
    hi_wilson: float
    verdict: str
    note: str = ""


class AuditReport(list):
    """A list of TailRatio rows with a readable printout."""

    def __str__(self):
        head = f"{'p':>8} {'tau_p':>9} {'k':>7} {'M':>8} {'rho_hat':>8}  {'95% interval':>17}  verdict"
        lines = [head, "-" * len(head)]
        for t in self:
            lines.append(f"{t.p:>8.0e} {t.tau:>9.3f} {t.k:>7d} {t.M:>8d} {t.rho:>8.2f}  "
                         f"[{t.lo:>6.2f}, {t.hi:>6.2f}]  {t.verdict}{'  ' + t.note if t.note else ''}")
        return "\n".join(lines)

    def to_records(self):
        return [asdict(t) for t in self]


def audit(r_model, r_reference, depths=(1e-2, 1e-3, 1e-4), z: float = 1.96,
          threshold_error: bool = True) -> AuditReport:
    """Audit model severities against held-out reference severities.

    Parameters
    ----------
    r_model : (M,) severities of samples drawn from the model.
    r_reference : (n,) severities of held-out reference data; defines tau_p.
    depths : tail probabilities p to audit. A depth needs roughly 100 / p
        reference points to be estimated empirically.
    threshold_error : include the quadrature term for estimating tau_p.
    """
    r_model = np.asarray(r_model, dtype=np.float64)
    r_ref = np.asarray(r_reference, dtype=np.float64)
    M, n_ref = len(r_model), len(r_ref)
    out = AuditReport()
    for p in depths:
        tau = float(np.quantile(r_ref, 1 - p))
        k = int((r_model >= tau).sum())
        rho = k / M / p
        lw, hw = (v / p for v in wilson(k, M, z))
        lo, hi = quadrature(rho, lw, hw, p, n_ref, z) if threshold_error else (lw, hw)
        note = ""
        if n_ref * p < 50:
            note = f"only {n_ref * p:.0f} reference exceedances"
        v = verdict(lo, hi)
        if k <= 1:
            cl, ch = (x / p for x in clopper_pearson(k, M))
            if verdict(cl, ch) != v:
                v, note = "no verdict", (note + "; " if note else "") + "k<=1, exact interval disagrees"
        out.append(TailRatio(p, tau, k, M, rho, lo, hi, lw, hw, v, note))
    return out
