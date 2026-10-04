"""Transports for a general severity functional.

A transport phi_{r -> r'} moves a sample x with R(x) = r to a point with
R(phi(x)) = r'. The whitened radius has a closed form (rescale the whitened
coordinates). For any differentiable R, two training-free constructions work:

* gradient flow   dx/ds = G^{-1} grad R / (grad R^T G^{-1} grad R),
  run for time r' - r. Along the flow dR/ds = 1, so the target is hit exactly,
  and the flow line through x is preserved (a "shape" coordinate).
* minimal displacement   argmin_y ||y - x||_G^2  s.t.  R(y) = r'.

With G equal to the whitening metric, both reduce to the radial rescaling of
the whitened radius (see tests/test_transport.py).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["gradient_flow", "minimal_displacement"]


def gradient_flow(x, r_target, R, grad_R, G_inv=None, steps: int = 30):
    """Integrate the normalized gradient flow with RK4 for time r_target - R(x).

    x : (d,) or (n, d);  R(X) -> (n,);  grad_R(X) -> (n, d);  G_inv : (d, d) or None.
    """
    X = np.atleast_2d(np.asarray(x, dtype=np.float64)).copy()
    r_t = np.broadcast_to(np.asarray(r_target, dtype=np.float64), (len(X),))
    h = (r_t - R(X)) / steps

    def field(Y):
        g = grad_R(Y)
        Gg = g if G_inv is None else g @ G_inv.T
        return Gg / np.sum(g * Gg, axis=1, keepdims=True)

    for _ in range(steps):
        k1 = field(X)
        k2 = field(X + 0.5 * h[:, None] * k1)
        k3 = field(X + 0.5 * h[:, None] * k2)
        k4 = field(X + h[:, None] * k3)
        X += (h / 6.0)[:, None] * (k1 + 2 * k2 + 2 * k3 + k4)
    return X if np.ndim(x) == 2 else X[0]


def minimal_displacement(x, r_target, R, G=None):
    """Closest point (in the G-norm) on the level set {R = r_target}."""
    x = np.asarray(x, dtype=np.float64)
    G = np.eye(len(x)) if G is None else G
    res = minimize(lambda y: (y - x) @ G @ (y - x), x, method="SLSQP",
                   jac=lambda y: 2 * G @ (y - x),
                   constraints=[{"type": "eq", "fun": lambda y: R(y[None])[0] - r_target}],
                   options=dict(ftol=1e-12, maxiter=500))
    return res.x
