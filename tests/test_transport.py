import numpy as np

import rarecal as rc


def _setup(seed=0, d=6):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    X = rng.standard_normal((20_000, d)) @ A.T + 3.0
    sev = rc.WhitenedRadius(d).fit(X)
    V, lam, m = sev.components_, sev.eigvals_, sev.mean_
    G = V @ np.diag(1 / lam) @ V.T                      # whitening metric
    R = lambda Y: sev(Y)
    grad = lambda Y: (sev.whiten(Y) / R(Y)[:, None]) @ (V / np.sqrt(lam)).T
    return rng, X, sev, G, R, grad


def test_gradient_flow_matches_radial_rescaling():
    rng, X, sev, G, R, grad = _setup()
    x = X[:5]
    r_t = sev(x) * rng.uniform(1.3, 2.0, size=5)
    closed = sev.transport(x, r_t)
    flow = rc.gradient_flow(x, r_t, R, grad, G_inv=np.linalg.inv(G), steps=40)
    assert np.allclose(flow, closed, atol=1e-6)
    assert np.allclose(R(flow), r_t, atol=1e-8)


def test_minimal_displacement_matches_radial_rescaling():
    rng, X, sev, G, R, grad = _setup(1)
    x = X[7]
    r_t = float(sev(x[None])[0] * 1.7)
    y = rc.minimal_displacement(x, r_t, R, G=G)
    assert np.allclose(y, sev.transport(x[None], np.array([r_t]))[0], atol=1e-4)


def test_gradient_flow_on_curved_severity_hits_target():
    R = lambda Y: Y[:, 0] + Y[:, 1] ** 2 / 4
    grad = lambda Y: np.stack([np.ones(len(Y)), Y[:, 1] / 2], 1)
    x = np.array([[0.3, -1.0], [1.0, 2.0]])
    y = rc.gradient_flow(x, np.array([4.0, 5.0]), R, grad, steps=60)
    assert np.allclose(R(y), [4.0, 5.0], atol=1e-8)
