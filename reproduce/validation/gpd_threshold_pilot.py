"""Pilot study of the adaptive threshold rule against fixed thresholds on synthetic families."""
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm, genpareto, kstest, chi2
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad

T0 = time.time()
NREP, B_BOOT, CHI90 = 30, 200, chi2.ppf(0.90, 1)
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]

def gpd_mle(y):
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return xi, sg

def llA(y, xi, sg):
    if sg <= 0: return -np.inf
    a = 1 + xi * y / sg
    if a.min() <= 0: return -np.inf
    if abs(xi) < 1e-9:
        return -len(y) * np.log(sg) - y.sum() / sg
    return -len(y) * np.log(sg) - (1 + 1/xi) * np.log(a).sum()

def profile_ci(y, zeta, tau_u, p_point):
    def sig_of(xi, p):
        if p >= zeta: return None
        if abs(xi) < 1e-9: return tau_u / np.log(zeta / p)
        dd = (p / zeta) ** (-xi) - 1.0
        return xi * tau_u / dd if dd != 0 else None
    def lstar(p):
        def neg(xi):
            s = sig_of(xi, p)
            return 1e9 if s is None or s <= 0 else -llA(y, xi, s)
        return -minimize_scalar(neg, bounds=(-0.95, 2.0), method="bounded",
                                options=dict(xatol=1e-4)).fun
    p_ref = p_point if p_point > 0 else zeta * 1e-4
    grid = p_ref * 10.0 ** np.linspace(-2, 2, 25)
    ls = np.array([lstar(p) for p in grid])
    k = int(np.argmax(ls)); lmax = ls[k]; lp = np.log10(grid[k])
    def g(logp): return 2 * (lmax - lstar(10 ** logp)) - CHI90
    lo, hi = 0.0, zeta
    for sgn in (-1, 1):
        b_ = lp
        for step in np.arange(0.25, 8, 0.25):
            b_ = lp + sgn * step
            if sgn > 0 and 10 ** b_ >= zeta * 0.999:
                b_ = np.log10(zeta * 0.999); break
            if g(b_) > 0: break
        try:
            root = brentq(g, min(lp, b_), max(lp, b_), xtol=1e-3)
            if sgn < 0: lo = 10 ** root
            else: hi = 10 ** root
        except ValueError:
            pass
    return lo, hi

def boot_ci(r, uq, tau, rng, B=B_BOOT):
    n = len(r); ps = []
    for _ in range(B):
        rb = r[rng.integers(0, n, n)]
        ub = np.quantile(rb, uq)
        y = rb[rb > ub] - ub
        if len(y) < 30: continue
        xi, sg = gpd_mle(y)
        ps.append((rb > ub).mean() * genpareto.sf(tau - ub, xi, 0.0, sg))
    ps = np.array(ps); ps = ps[np.isfinite(ps)]
    return np.quantile(ps, 0.05), np.quantile(ps, 0.95)

def fit_at(r, uq, tau, p_true, rng):
    u = np.quantile(r, uq)
    y = r[r > u] - u
    xi, sg = gpd_mle(y)
    zeta = (r > u).mean()
    ph = zeta * genpareto.sf(tau - u, xi, 0.0, sg)
    ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
    lo_p, hi_p = profile_ci(y, zeta, tau - u, ph)
    lo_b, hi_b = boot_ci(r, uq, tau, rng)
    return dict(ratio=ph / p_true, zero=float(ph == 0), ksp=ksp, q=uq,
                cp=float(lo_p <= p_true <= hi_p),
                cb=float(lo_b <= p_true <= hi_b),
                wp=np.log10(hi_p / lo_p) if lo_p > 0 else np.nan,
                wb=np.log10(hi_b / lo_b) if lo_b > 0 else np.nan)

def run_example(name, draw_r, tau, p_true, seed0):
    per_q = {q: [] for q in QGRID}
    adapt = []
    for k in range(NREP):
        rg = np.random.default_rng(seed0 + k)
        r = draw_r(rg)
        fits = [fit_at(r, q, tau, p_true, rg) for q in QGRID]
        for q, f in zip(QGRID, fits):
            per_q[q].append(f)
        # adaptive: max KS p, ties (within 1e-12) -> deeper
        ks = np.array([f["ksp"] for f in fits])
        adapt.append(fits[int(np.argmax(ks + 1e-12 * np.arange(len(QGRID))))])
    def show(tag, rows):
        rat = np.array([f["ratio"] for f in rows])
        print(f"  [{name} {tag:>6}] ratio med={np.median(rat):.2f} "
              f"IQR=[{np.quantile(rat,.25):.2f},{np.quantile(rat,.75):.2f}] "
              f"zero={np.mean([f['zero'] for f in rows]):.2f} | "
              f"prof cov={np.mean([f['cp'] for f in rows]):.2f} "
              f"w={np.nanmedian([f['wp'] for f in rows]):.1f} | "
              f"boot cov={np.mean([f['cb'] for f in rows]):.2f} "
              f"w={np.nanmedian([f['wb'] for f in rows]):.1f} | "
              f"med q={np.median([f['q'] for f in rows]):.4f}"
              f"   ({time.time()-T0:.0f}s)", flush=True)
    for q in QGRID:
        show(f"{q}", per_q[q])
    show("adapt", adapt)

# --- ex1: linear severity, Gaussian
run_example("ex1", lambda rg: rg.standard_normal(50_000),
            4.8, norm.sf(4.8), 10_000)

# --- ex2: radial severity, elliptical t3 (tau fixed by one big pass, as before)
L = np.linalg.cholesky(np.array([[1, .6], [.6, 1]]))
def draw_t(k, rg):
    z = rg.standard_normal((k, 2)) @ L.T
    return z / np.sqrt(rg.chisquare(3.0, k) / 3.0)[:, None]
rgT = np.random.default_rng(77)
pre = np.linalg.norm(draw_t(10_000_000, rgT), axis=1)
tau_r = np.quantile(pre, 1 - 3e-6)
allR = []
for _ in range(20):
    allR.append(np.linalg.norm(draw_t(5_000_000, rgT), axis=1))
allR = np.concatenate([a[a >= tau_r] for a in allR])
TAU2 = float(np.sort(allR)[-100]); P2 = 100 / 1e8
print(f"ex2 tau={TAU2:.1f}  p_true={P2:.1e}  ({time.time()-T0:.0f}s)", flush=True)
run_example("ex2", lambda rg: np.linalg.norm(draw_t(50_000, rg), axis=1),
            TAU2, P2, 20_000)

# --- ex3: curved severity
TAU3 = brentq(lambda t: quad(lambda w: norm.pdf(w) * norm.sf(t - w*w/4), -30, 30)[0]
              - 1e-5, 3, 40)
def draw3(rg):
    X = rg.standard_normal((50_000, 2))
    return X[:, 0] + X[:, 1] ** 2 / 4
run_example("ex3", draw3, TAU3, 1e-5, 30_000)

# --- ex4: d=10 equicorrelated Gaussian, mean severity
D = 10
SIG = 0.5 * np.eye(D) + 0.5
L10 = np.linalg.cholesky(SIG)
sdR = np.sqrt((1 + (D - 1) * 0.5) / D)
TAU4 = sdR * norm.isf(1e-5)
run_example("ex4", lambda rg: (rg.standard_normal((50_000, D)) @ L10.T).mean(1),
            TAU4, 1e-5, 40_000)

print("DONE", flush=True)
