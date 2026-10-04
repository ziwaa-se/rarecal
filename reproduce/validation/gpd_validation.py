"""Validation of the calibration law's generalized-Pareto tail on synthetic families with known tail probabilities."""
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm, genpareto, spearmanr, kstest, chi2
from scipy.optimize import minimize, brentq, minimize_scalar
from scipy.integrate import quad

T0 = time.time()
NREP, B_BOOT, CHI90 = 100, 400, chi2.ppf(0.90, 1)
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

def shape_fit(rB, psiB):
    def nl(p):
        a, lc, b = p
        s = np.exp(lc) * rB ** b
        return np.sum(np.log(s) + 0.5 * ((psiB - a * rB) / s) ** 2)
    st = min((minimize(nl, x0, method="BFGS")
              for x0 in ([0, 0, 0.3], [0.5, 0, 0.1], [0, 0.5, 0.6])),
             key=lambda q: q.fun).x
    a, c, b = st[0], np.exp(st[1]), st[2]
    return a, c, b, (psiB - a * rB) / (c * rB ** b)

def select_u(r):
    """Data-only threshold choice: max KS p over the grid, ties -> deeper."""
    best, best_score = None, -1.0
    for j, q in enumerate(QGRID):
        u = np.quantile(r, q)
        y = r[r > u] - u
        xi, sg = gpd_mle(y)
        ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
        score = ksp + 1e-12 * j
        if score > best_score:
            best_score, best = score, (q, u, y, xi, sg)
    return best

def one_rep(r, psi, tau, p_true, uqB, rng):
    uq, u, y, xi, sg = select_u(r)
    ph = (r > u).mean() * genpareto.sf(tau - u, xi, 0.0, sg)
    zeta = (r > u).mean()
    lo_p, hi_p = profile_ci(y, zeta, tau - u, ph)
    lo_b, hi_b = boot_ci(r, uq, tau, rng)
    mB = r >= np.quantile(r, uqB)
    _, _, _, zsh = shape_fit(r[mB], psi[mB])
    E = np.log1p(xi * y / sg) / xi if abs(xi) > 1e-9 else y / sg
    ks = kstest(E, "expon").pvalue
    ind = spearmanr(np.abs(zsh), r[mB])[1]
    return dict(ratio=ph / p_true, zero=float(ph == 0), q=uq,
                cb=float(lo_b <= p_true <= hi_b), cp=float(lo_p <= p_true <= hi_p),
                wb=np.log10(hi_b / lo_b) if lo_b > 0 else np.nan,
                wp=np.log10(hi_p / lo_p) if lo_p > 0 else np.nan,
                ks=float(ks < 0.05), ind=float(ind < 0.05))

def summarize(name, rows):
    R = {k: np.array([d_[k] for d_ in rows]) for k in rows[0]}
    rat = R["ratio"]
    print(f"[{name}] ratio med={np.median(rat):.2f} "
          f"IQR=[{np.quantile(rat,.25):.2f},{np.quantile(rat,.75):.2f}] "
          f"zero={R['zero'].mean():.2f} | cover boot={R['cb'].mean():.2f} "
          f"prof={R['cp'].mean():.2f} | width(dec) boot={np.nanmedian(R['wb']):.1f} "
          f"prof={np.nanmedian(R['wp']):.1f} | rej KS={R['ks'].mean():.2f} "
          f"ind={R['ind'].mean():.2f} | med q={np.median(R['q']):.4f}"
          f"   ({time.time()-T0:.0f}s)", flush=True)

# --- ex1
rows = []
for k in range(NREP):
    rg = np.random.default_rng(10_000 + k)
    r = rg.standard_normal(50_000)
    psi = 0.8 * r + 0.6 * rg.standard_normal(50_000)
    rows.append(one_rep(r, psi, 4.8, norm.sf(4.8), 0.97, rg))
summarize("ex1 linear/Gauss", rows)

# --- ex2
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
rows = []
for k in range(NREP):
    rg = np.random.default_rng(20_000 + k)
    X = draw_t(50_000, rg)
    r = np.linalg.norm(X, axis=1)
    th = np.arctan2(X[:, 1], X[:, 0])
    dj = np.angle(np.exp(1j * (th - np.pi/4)))
    dk = np.angle(np.exp(1j * (th - np.pi/4 - np.pi)))
    delta = np.where(np.abs(dj) < np.abs(dk), dj, dk)
    rows.append(one_rep(r, delta, TAU2, P2, 0.985, rg))
summarize("ex2 radial/t3   ", rows)

# --- ex3
TAU3 = brentq(lambda t: quad(lambda w: norm.pdf(w) * norm.sf(t - w*w/4), -30, 30)[0]
              - 1e-5, 3, 40)
def gradR3(Y):
    g = np.ones_like(Y); g[:, 1] = Y[:, 1] / 2
    return g
def transp3(Y, rf, rt, steps=30):
    y = Y.astype(np.float64).copy()
    h = (np.asarray(rt, float) - rf) / steps
    for _ in range(steps):
        f = lambda w: gradR3(w) / (gradR3(w) ** 2).sum(1, keepdims=True)
        k1 = f(y); k2 = f(y + .5*h[:, None]*k1)
        k3 = f(y + .5*h[:, None]*k2); k4 = f(y + h[:, None]*k3)
        y += (h/6)[:, None] * (k1 + 2*k2 + 2*k3 + k4)
    return y
rows = []
for k in range(NREP):
    rg = np.random.default_rng(30_000 + k)
    X = rg.standard_normal((50_000, 2))
    r = X[:, 0] + X[:, 1] ** 2 / 4
    mB = r >= np.quantile(r, 0.97)
    psi = np.zeros_like(r)
    psi[mB] = transp3(X[mB], r[mB], np.full(mB.sum(), np.median(r[mB])))[:, 1]
    rows.append(one_rep(r, psi, TAU3, 1e-5, 0.97, rg))
summarize("ex3 curved      ", rows)

# --- ex4
D = 10
SIG = 0.5 * np.eye(D) + 0.5
L10 = np.linalg.cholesky(SIG)
sdR = np.sqrt((1 + (D - 1) * 0.5) / D)
TAU4 = sdR * norm.isf(1e-5)
V10 = np.linalg.svd(np.ones((1, D)))[2][1:].T
rows = []
for k in range(NREP):
    rg = np.random.default_rng(40_000 + k)
    X = rg.standard_normal((50_000, D)) @ L10.T
    r = X.mean(1)
    psi = (X @ V10)[:, 0]
    rows.append(one_rep(r, psi, TAU4, 1e-5, 0.97, rg))
summarize("ex4 d=10 mean   ", rows)
print("DONE", flush=True)
