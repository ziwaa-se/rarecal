"""CelebA-HQ-256: reduce to 32x32, fit R1 and the calibration law, and self-validate."""
import glob
import io
import os
import sys
import time
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc  # noqa: E402

T0 = time.time()
P3 = os.path.expandvars("${CACHE_ROOT}/paper")
DATA = os.path.expandvars("${PROJECT_ROOT}/paper/data")
REAL32 = f"{P3}/d2_real32.npz"
LAYER = f"{P3}/d2_layer_seed8000.npz"
D = 3072
NFIT, SEED = 10_000, 8000
K = 50
DEPTHS = [1e-2, 3e-3, 1e-3]
Z = 1.96
torch.set_num_threads(8)

os.makedirs(DATA, exist_ok=True)


def quad_ci(rho, lo, hi, p, n_h, z=Z):
    """Quadrature certificate (CONDITION 1; numerics = self_validation_resplit.py)."""
    sig_thr = np.sqrt((1 - p) / (n_h * p)) * max(rho, 1e-9)
    add = z * sig_thr
    return (rho - np.sqrt((rho - lo) ** 2 + add ** 2),
            rho + np.sqrt((hi - rho) ** 2 + add ** 2))


# ---------------- 1. corpus -> 32x32 uint8 (cached) ----------------
if os.path.exists(REAL32):
    X32 = np.load(REAL32)["X32"]
    print(f"loaded corpus cache: {X32.shape} ({time.time()-T0:.0f}s)",
          flush=True)
else:
    import pandas as pd
    from PIL import Image
    shards = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--korexyz--celeba-hq-256x256/"
        "snapshots/*/data/*.parquet")))
    print(f"{len(shards)} shards (sorted glob order, deterministic)",
          flush=True)
    imgs = []
    for sh in shards:
        df = pd.read_parquet(sh, columns=["image"])
        for b in df["image"]:
            im = np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"),
                            dtype=np.float32)              # (256,256,3)
            x32 = im.reshape(32, 8, 32, 8, 3).mean(axis=(1, 3))  # 8x box mean
            imgs.append(np.clip(np.round(x32), 0, 255).astype(np.uint8))
        print(f"  {os.path.basename(sh)}: total {len(imgs)} "
              f"({time.time()-T0:.0f}s)", flush=True)
    X32 = np.stack(imgs)
    np.savez_compressed(REAL32, X32=X32)
    print(f"corpus cached: {X32.shape} -> {REAL32} ({time.time()-T0:.0f}s)",
          flush=True)

N = len(X32)
perm = np.random.default_rng(SEED).permutation(N)   # seed-8000 recipe
fit_idx, hold_idx = perm[:NFIT], perm[NFIT:]
n_h = len(hold_idx)
print(f"split: {len(fit_idx)} fit / {n_h} hold (seed {SEED})", flush=True)

# ---------------- 2. R1 basis (severity build_basis numerics, N=30k) -------
Xf = torch.tensor(X32[np.sort(fit_idx)].reshape(-1, D).astype(np.float32))
m = Xf.double().mean(0).float()
C = torch.zeros((D, D), dtype=torch.float32)
for a in range(0, len(Xf), 5_000):
    ch = Xf[a:a + 5_000] - m
    C += ch.T @ ch
C /= len(Xf)
lam, V = torch.linalg.eigh(C.double())
lam = lam[-K:].flip(0).float()
V = V[:, -K:].flip(1).float()
print(f"basis built: top-{K} eigenvalues [{lam[-1]:.1f}..{lam[0]:.1f}] "
      f"({time.time()-T0:.0f}s)", flush=True)

Wmat = V / torch.sqrt(lam)
Wall = np.empty((N, K), dtype=np.float32)
for a in range(0, N, 10_000):
    ch = torch.tensor(X32[a:a + 10_000].reshape(-1, D).astype(np.float32)) - m
    Wall[a:a + 10_000] = (ch @ Wmat).numpy()
R50 = np.sqrt((Wall.astype(np.float64) ** 2).sum(1))     # sc.radius_k(., 50)
Rf, Rh = R50[fit_idx], R50[hold_idx]

# ---------------- 3. GPD layer: adaptive-KS threshold (identical) ----------
q, u, xi, sg, ksp = sc.select_u(Rf)
zeta = float((Rf > u).mean())
taus = {p: float(np.quantile(Rh, 1 - p)) for p in DEPTHS}
Rf_sorted = np.sort(Rf)
tau4 = float(sc.spliced_quantile(np.array([1 - 1e-4]), Rf_sorted,
                                 u, xi, sg, zeta)[0])   # GPD extrapolation
print(f"[CelebA-HQ R1 calibration law] q={q} u={u:.3f} xi={xi:+.3f} sg={sg:.3f} ksp={ksp:.3f} "
      f"zeta={zeta:.4f} n_exc={int((Rf > u).sum())}", flush=True)
print("  taus: " + "  ".join(f"{p:g}:{t:.3f}" for p, t in taus.items()) +
      f"  1e-4(GPD):{tau4:.3f}", flush=True)

np.savez(LAYER, m=m.numpy(), V=V.numpy(), lam=lam.numpy(), R50=R50,
         fit_idx=fit_idx, hold_idx=hold_idx, q=q, u=u, xi=xi, sg=sg,
         ksp=ksp, zeta=zeta, tau2=taus[1e-2], tau3e3=taus[3e-3],
         tau3=taus[1e-3], tau4=tau4)

with open(f"{DATA}/celebahq_calibration_law.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["functional", "n_fit", "n_hold", "q", "u", "xi", "sigma",
                "ks_p", "zeta", "n_excess", "p", "tau", "tau_source",
                "thr_rel_err_delta"])
    for p in DEPTHS:
        w.writerow(["S1_pca50", NFIT, n_h, q, f"{u:.6f}", f"{xi:.6f}",
                    f"{sg:.6f}", f"{ksp:.6f}", f"{zeta:.6f}",
                    int((Rf > u).sum()), p, f"{taus[p]:.6f}",
                    "hold_empirical", f"{np.sqrt((1-p)/(n_h*p)):.4f}"])
    w.writerow(["S1_pca50", NFIT, n_h, q, f"{u:.6f}", f"{xi:.6f}",
                f"{sg:.6f}", f"{ksp:.6f}", f"{zeta:.6f}",
                int((Rf > u).sum()), 1e-4, f"{tau4:.6f}",
                "gpd_extrapolation_FLAGGED", ""])
print(f"wrote celebahq_calibration_law.csv ({time.time()-T0:.0f}s)", flush=True)

# ---------------- 4. self-validation (fit in the model role) ---------------
rows = []
for p in DEPTHS:
    a = sc.audit_rho(Rf, taus[p], p)
    qlo, qhi = quad_ci(a["rho"], a["lo"], a["hi"], p, n_h)
    rows.append(dict(p=p, tau=f"{taus[p]:.6f}", n_fit=a["M"], k=a["k"],
                     rho_self=f"{a['rho']:.4f}",
                     wilson_lo=f"{a['lo']:.4f}", wilson_hi=f"{a['hi']:.4f}",
                     quad_lo=f"{qlo:.4f}", quad_hi=f"{qhi:.4f}",
                     covers1_wilson=int(a["lo"] <= 1.0 <= a["hi"]),
                     covers1_quad=int(qlo <= 1.0 <= qhi)))
    print(f"self-val p={p:g}: rho_self={a['rho']:.3f} k={a['k']} "
          f"wilson=[{a['lo']:.3f},{a['hi']:.3f}] quad=[{qlo:.3f},{qhi:.3f}]",
          flush=True)

# corrected self-error e(p) = (1 - F_hat(tau_p))/p (CONDITION 4 definition)
def e_inherited(tau_p, p):
    if tau_p > u:
        return zeta * max(1 + xi * (tau_p - u) / sg, 0.0) ** (-1 / xi) / p
    return float((Rf > tau_p).mean()) / p

tau4_hold = float(np.quantile(Rh, 1 - 1e-4))   # near sample max: disclosed
e_rows = [
    dict(p=1e-3, tau=f"{taus[1e-3]:.6f}", tau_source="hold_empirical",
         e_self=f"{e_inherited(taus[1e-3], 1e-3):.4f}"),
    dict(p=1e-4, tau=f"{tau4_hold:.6f}",
         tau_source="hold_empirical_NEAR_MAX_disclosed",
         e_self=f"{e_inherited(tau4_hold, 1e-4):.4f}"),
    dict(p=1e-4, tau=f"{tau4:.6f}", tau_source="gpd_extrapolation",
         e_self=f"{e_inherited(tau4, 1e-4):.4f}"),  # =1 by construction
]
for r in e_rows:
    print(f"self-error e({r['p']:g}) [{r['tau_source']}] = {r['e_self']}",
          flush=True)

with open(f"{DATA}/celebahq_self_validation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{DATA}/celebahq_self_error.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(e_rows[0].keys()))
    w.writeheader(); w.writerows(e_rows)
print(f"wrote celebahq_self_validation.csv, celebahq_self_error.csv  DONE "
      f"({time.time()-T0:.0f}s)", flush=True)
