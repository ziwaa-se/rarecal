"""Fit the R1 functional and the calibration law for the wrapping/FID experiment."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import LAYER, LAYER_DIR, PAPER_LAYER, N_HOLD_PAPER      # noqa: E402
import severity as sc                                              # noqa: E402

T0 = time.time()
SEED = 8000
os.makedirs(LAYER_DIR, exist_ok=True)

b = sc.build_basis(SEED, kmax=sc.K)
Wall = sc.project_reals(b)
R50 = sc.radius_k(Wall, sc.K)
fit_idx, hold_idx = b["fit_idx"], b["hold_idx"]
Rf, Rh = R50[fit_idx], R50[hold_idx]
q, u, xi, sg, ksp = sc.select_u(Rf)
zeta = float((Rf > u).mean())
tau3 = float(np.quantile(Rh, 1 - 1e-3))
tau4 = float(np.quantile(Rh, 1 - 1e-4))
print(f"layer: q={q} u={u:.6f} xi={xi:+.6f} sg={sg:.6f} ksp={ksp:.4f} zeta={zeta:.4f} "
      f"tau3={tau3:.6f} tau4={tau4:.6f}  n_fit={len(fit_idx)} n_hold={len(hold_idx)}  "
      f"({time.time()-T0:.0f}s)", flush=True)

np.savez(LAYER, seed=SEED, m=b["m"].numpy(), V=b["V"].numpy(), lam=b["lam"].numpy(),
         fit_idx=fit_idx, hold_idx=hold_idx, N=b["N"], R50=R50,
         q=q, u=u, xi=xi, sg=sg, ksp=ksp, zeta=zeta, tau3=tau3, tau4=tau4)
print(f"wrote {LAYER}", flush=True)

# ---- check against the reference constants
ours = dict(q=q, u=u, xi=xi, sg=sg, zeta=zeta, tau3=tau3, tau4=tau4)
ok = True
for k, v in PAPER_LAYER.items():
    d = abs(ours[k] - v)
    flag = "ok" if d < 1e-3 else "MISMATCH"
    ok &= d < 1e-3
    print(f"  {k:5s} ours={ours[k]:.6f} ref={v:.6f} |diff|={d:.2e} {flag}", flush=True)
print(f"  n_hold ours={len(hold_idx)} ref={N_HOLD_PAPER} "
      f"{'ok' if len(hold_idx) == N_HOLD_PAPER else 'MISMATCH'}", flush=True)
print("LAYER MATCHES REFERENCE" if ok else "LAYER DOES NOT MATCH REFERENCE (corpus or numerics differ; "
      "the experiment is still valid on our own layer, but Table-1 rho values will not be reproduced)",
      flush=True)

# ---- optional: bitwise cross-check with the advisor's readable layer
try:
    z = np.load(os.path.expandvars("${CACHE_ROOT}/validation/layer_seed8000.npz"))
    d = np.abs(z["R50"] - R50)
    print(f"  [optional] vs advisor earlier layer: max|R50 diff|={d.max():.3e}, "
          f"tau3 diff={abs(float(z['tau3'])-tau3):.3e}", flush=True)
except Exception as e:
    print(f"  [optional] advisor layer not readable ({type(e).__name__}); skipped", flush=True)
print("DONE", flush=True)
