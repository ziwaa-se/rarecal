import numpy as np
import pytest

import rarecal as rc


def student_t(n, d, df, rng):
    """Multivariate Student-t rescaled to identity covariance."""
    z = rng.standard_normal((n, d))
    g = rng.chisquare(df, size=(n, 1)) / df
    return z / np.sqrt(g) * np.sqrt((df - 2) / df)


def test_wilson_matches_known_values():
    lo, hi = rc.wilson(1, 100_000)
    assert lo / 1e-4 == pytest.approx(0.0177, abs=1e-3)
    assert hi / 1e-4 == pytest.approx(0.566, abs=2e-3)
    wl, wh = rc.wald(1, 100_000)
    assert wl < 0 < wh                       # Wald goes negative at k = 1


def test_clopper_pearson_upper_limit():
    lo, hi = rc.clopper_pearson(1, 100_000)
    assert hi / 1e-4 == pytest.approx(0.557, abs=1e-3)


def test_audit_is_calibrated_on_real_vs_real():
    rng = np.random.default_rng(0)
    X = student_t(300_000, 8, 6, rng)
    sev = rc.WhitenedRadius(8).fit(X[:100_000])
    rep = rc.audit(sev(X[100_000:200_000]), sev(X[200_000:]), depths=[1e-2, 1e-3])
    assert all(t.verdict == "calibrated" for t in rep)


def test_audit_detects_light_tail():
    rng = np.random.default_rng(1)
    X = student_t(300_000, 8, 5, rng)
    sev = rc.WhitenedRadius(8).fit(X[:100_000])
    gauss = rng.standard_normal((100_000, 8))       # same covariance, lighter tail
    rep = rc.audit(sev(gauss), sev(X[200_000:]), depths=[1e-3])
    assert rep[0].verdict == "under" and rep[0].rho < 0.5


def test_wrapper_repairs_the_rate_and_keeps_directions():
    rng = np.random.default_rng(2)
    X = student_t(300_000, 8, 5, rng)
    sev = rc.WhitenedRadius(8).fit(X[:100_000])
    law = rc.CalibrationLaw.fit(sev(X[:100_000]))
    gauss = rng.standard_normal((100_000, 8))
    Xw = rc.Wrapper(sev, law, tail_only=0.99)(gauss)
    rep = rc.audit(sev(Xw), sev(X[200_000:]), depths=[1e-3])
    assert rep[0].verdict == "calibrated"
    w0, w1 = sev.whiten(gauss), sev.whiten(Xw)
    cos = np.sum(w0 * w1, 1) / np.linalg.norm(w0, axis=1) / np.linalg.norm(w1, axis=1)
    assert np.allclose(cos, 1.0)                     # direction (shape) preserved


def test_calibration_law_quantile_inverts_sf():
    rng = np.random.default_rng(3)
    law = rc.CalibrationLaw.fit(rng.standard_exponential(200_000))
    v = np.array([0.5, 0.99, 0.999, 0.9999])
    assert np.allclose(law.cdf(law.ppf(v)), v, atol=2e-5)


def test_sample_size_bound_arithmetic():
    assert round(rc.sample_size_lower_bound(1e-4, 2)) == 3238
    assert round(rc.sample_size_lower_bound(1e-3, 2)) == 324
