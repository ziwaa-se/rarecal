"""Fit the R1 functional and the calibration law on the calibration split and read the audited thresholds from the held-out split."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

t0 = time.time()
os.makedirs(sc.P2, exist_ok=True)

EXPECT_N3 = {"iDDPM": 17, "EDM": 181, "EDM-churn": 30, "EDM2-S": 147,
             "EDM2-M": 82, "StyleGAN-XL": 103}

# ---------------- seed 8000: full K100 layer + model coords ----------------
b = sc.build_basis(8000, kmax=100)
Wall = sc.project_reals(b)
np.save(f"{sc.P2}/wall100_seed8000.npy", Wall)
R50 = sc.radius_k(Wall, 50)
Rf, Rh = R50[b["fit_idx"]], R50[b["hold_idx"]]
q, u, xi, sg, ksp = sc.select_u(Rf)
tau3, tau4 = np.quantile(Rh, 1 - 1e-3), np.quantile(Rh, 1 - 1e-4)
print(f"[seed 8000 K50] q={q} xi={xi:+.3f} sg={sg:.3f} ksp={ksp:.3f} "
      f"tau3={tau3:.3f} tau4={tau4:.3f}  (expect q=0.9925 xi=-0.050 sg=0.974 "
      f"tau4=16.13)", flush=True)
np.savez(f"{sc.P2}/layer_seed8000.npz", m=b["m"].numpy(), V=b["V"].numpy(),
         lam=b["lam"].numpy(), R50=R50, q=q, u=u, xi=xi, sg=sg, ksp=ksp,
         zeta=(Rf > u).mean(), tau3=tau3, tau4=tau4)

for name, path in sc.CONFIGS:
    X = sc.load_cache(path)
    W = sc.project_images(X, b)
    np.savez(f"{sc.P2}/model_w100_{sc.SLUG[name]}.npz", W=W)
    Rm = sc.radius_k(W, 50)
    n3 = int((Rm >= tau3).sum())
    exp = EXPECT_N3.get(name)
    flag = "" if exp is None else ("  OK" if n3 == exp else f"  MISMATCH exp={exp}")
    print(f"  {name:12s} M={len(Rm):6d} n3={n3}{flag} n4={int((Rm>=tau4).sum())} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del X, W

# ---------------- seeds 8001-8004: K50 layers; models for 8001/8002 --------
for seed in [8001, 8002, 8003, 8004]:
    bs = sc.build_basis(seed, kmax=50)
    Ws = sc.project_reals(bs)
    R = sc.radius_k(Ws, 50)
    del Ws
    Rfs = R[bs["fit_idx"]]
    qs, us, xis, sgs, ksps = sc.select_u(Rfs)
    np.savez(f"{sc.P2}/layer_seed{seed}.npz", m=bs["m"].numpy(),
             V=bs["V"].numpy(), lam=bs["lam"].numpy(), R50=R, q=qs, u=us,
             xi=xis, sg=sgs, ksp=ksps, zeta=(Rfs > us).mean(),
             tau3=np.quantile(R[bs["hold_idx"]], 1 - 1e-3),
             tau4=np.quantile(R[bs["hold_idx"]], 1 - 1e-4))
    print(f"[seed {seed} K50] q={qs} xi={xis:+.3f} sg={sgs:.3f} ksp={ksps:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if seed in (8001, 8002):
        for name, path in sc.CONFIGS:
            X = sc.load_cache(path)
            W = sc.project_images(X, bs)
            np.save(f"{sc.P2}/model_r50_seed{seed}_{sc.SLUG[name]}.npy",
                    sc.radius_k(W, 50))
            del X, W
        print(f"[seed {seed}] model radii saved ({time.time()-t0:.0f}s)",
              flush=True)

print(f"ALL LAYERS BUILT ({time.time()-t0:.0f}s)", flush=True)
