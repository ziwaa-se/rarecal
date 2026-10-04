"""Sample EDM (ImageNet-64, deterministic Heun) and audit it under R1."""
import os
import sys
import time
import pickle
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest

T0 = time.time()
WALL_GUARD = 3.2 * 3600
EDMDIR = os.path.expandvars("${PROJECT_ROOT}/external/edm")
PKL = os.path.expandvars("${PROJECT_ROOT}/external/edm-imagenet-64x64-cond-adm.pkl")
NPY = os.path.expandvars("${DATA_ROOT}/imagenet32_train.npy")
CACHE = os.path.expandvars("${PROJECT_ROOT}/docs/edm_samples_cache.npz")
sys.path.insert(0, EDMDIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K, NFIT = 50, 200_000
QGRID = [0.98, 0.985, 0.99, 0.9925, 0.995]
M0, NKEEP, BS = 100_000, 1000, 256
INK, SURFACE = "#33322e", "#fcfcfb"

X8 = np.load(NPY, mmap_mode="r")
N = X8.shape[0]
D = 32 * 32 * 3
print(f"{N:,} real images, device={DEVICE}  ({time.time()-T0:.0f}s)", flush=True)

# ---------------------------------------------------------------- severity machinery
def gpd_mle(y):
    xi, _, sg = genpareto.fit(y, floc=0.0)
    return xi, sg

def select_u(r):
    best, best_score = None, -1.0
    for j, q in enumerate(QGRID):
        u = np.quantile(r, q)
        y = r[r > u] - u
        xi, sg = gpd_mle(y)
        ksp = kstest(y, "genpareto", args=(xi, 0.0, sg)).pvalue
        if ksp + 1e-12 * j > best_score:
            best_score, best = ksp + 1e-12 * j, (q, u, y, xi, sg)
    return best

rng = np.random.default_rng(8000)
perm = rng.permutation(N)
fit_idx, hold_idx = perm[:NFIT], perm[NFIT:]
msum = torch.zeros(D, dtype=torch.float64, device=DEVICE)
for a in range(0, len(fit_idx), 50_000):
    ch = torch.tensor(np.ascontiguousarray(
        X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
        dtype=torch.float32, device=DEVICE)
    msum += ch.sum(0).double()
m = (msum / len(fit_idx)).float()
C = torch.zeros((D, D), dtype=torch.float32, device=DEVICE)
for a in range(0, len(fit_idx), 50_000):
    ch = torch.tensor(np.ascontiguousarray(
        X8[np.sort(fit_idx[a:a+50_000])].reshape(-1, D)),
        dtype=torch.float32, device=DEVICE) - m
    C += ch.T @ ch
C /= len(fit_idx)
lam, V = torch.linalg.eigh(C.double())
lam, V = lam[-K:].flip(0).float(), V[:, -K:].flip(1).float()
Wmat = V / torch.sqrt(lam)
R = np.empty(N, dtype=np.float64)
Wall = np.empty((N, K), dtype=np.float32)
for a in range(0, N, 100_000):
    ch = torch.tensor(np.ascontiguousarray(X8[a:a+100_000].reshape(-1, D)),
                      dtype=torch.float32, device=DEVICE) - m
    w = ch @ Wmat
    R[a:a+100_000] = torch.linalg.norm(w, dim=1).cpu().numpy()
    Wall[a:a+100_000] = w.cpu().numpy()
Rf, Rh = R[fit_idx], R[hold_idx]
uq, u, y, xi, sg = select_u(Rf)
zeta = (Rf > u).mean()
tau = np.quantile(Rh, 1 - 1e-4)
tau3 = np.quantile(Rh, 1 - 1e-3)
print(f"severity layer: q={uq} xi={xi:+.3f} sg={sg:.3f}  tau(1e-4)={tau:.2f}  "
      f"({time.time()-T0:.0f}s)", flush=True)

# ---------------------------------------------------------------- EDM sampling
def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like,
                num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    # verbatim from NVlabs/edm generate.py (deterministic when S_churn=0)
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

done = 0
X32 = np.zeros((M0, D), dtype=np.uint8)
if os.path.exists(CACHE):
    cc = np.load(CACHE)
    prev = cc["X32"]
    done = min(len(prev), M0)
    X32[:done] = prev[:done]
    print(f"resumed cache: {done} samples", flush=True)

if done < M0:
    with open(PKL, "rb") as f:
        net = pickle.load(f)["ema"].to(DEVICE)
    net.eval()
    print(f"loaded EDM net (label_dim={net.label_dim}, res={net.img_resolution})  "
          f"({time.time()-T0:.0f}s)", flush=True)
    gt = torch.Generator(device=DEVICE)
    gt.manual_seed(1234 + done)
    eye = torch.eye(net.label_dim, device=DEVICE)
    t_last = time.time()
    while done < M0 and time.time() - T0 < WALL_GUARD:
        nb = min(BS, M0 - done)
        lat = torch.randn((nb, net.img_channels, net.img_resolution,
                           net.img_resolution), device=DEVICE, generator=gt)
        cls = eye[torch.randint(net.label_dim, (nb,), device=DEVICE, generator=gt)]
        with torch.no_grad():
            x = edm_sampler(net, lat, cls)
        px = (x * 127.5 + 128).clip(0, 255)                     # (nb,3,64,64)
        px32 = torch.nn.functional.avg_pool2d(px.float(), 2)    # box 2x2 -> 32
        hwc = px32.permute(0, 2, 3, 1).round().clip(0, 255).to(torch.uint8)
        X32[done:done+nb] = hwc.reshape(nb, D).cpu().numpy()
        done += nb
        if done % 12_800 < BS:
            print(f"  {done}/{M0} sampled ({(done)/(time.time()-t_last+1e-9):.1f} "
                  f"img/s cum)  ({time.time()-T0:.0f}s)", flush=True)
            np.savez_compressed(CACHE, X32=X32[:done])
    np.savez_compressed(CACHE, X32=X32[:done])
    print(f"sampling finished/paused at {done}  ({time.time()-T0:.0f}s)", flush=True)

M = done
Xm = X32[:M].astype(np.float32)
Wg = np.empty((M, K), dtype=np.float32)
for a in range(0, M, 50_000):
    ch = torch.tensor(Xm[a:a+50_000], dtype=torch.float32, device=DEVICE) - m
    Wg[a:a+50_000] = (ch @ Wmat).cpu().numpy()
Rm = np.linalg.norm(Wg, axis=1)
print(f"EDM own severity-tail rate ({M} samples): at tau(1e-3) "
      f"{(Rm >= tau3).mean():.2e} (truth 1e-3, n={(Rm >= tau3).sum()}), "
      f"at tau(1e-4) {(Rm >= tau).mean():.2e} (truth 1e-4, n={(Rm >= tau).sum()})",
      flush=True)
print(f"EDM severity quantiles q50/99/999: "
      f"{np.quantile(Rm,[.5,.99,.999]).round(2)}  real fit-split: "
      f"{np.quantile(Rf,[.5,.99,.999]).round(2)}", flush=True)

# ---------------------------------------------------------------- transport + validate
order = np.argsort(Rm)[-NKEEP:]
r_src = Rm[order]
print(f"kept top {NKEEP} of {M} (model-tail severities "
      f"[{r_src.min():.2f}, {r_src.max():.2f}])", flush=True)
Ee = rng.exponential(size=NKEEP)
st = sg + xi * (tau - u)
r_g = np.sort(tau + st * np.expm1(xi * Ee) / xi)
Vn, ln = V.cpu().numpy(), np.sqrt(lam.cpu().numpy())
Xt = Xm[order].astype(np.float64)
mt = m.cpu().numpy().astype(np.float64)
Wm_np = (V / torch.sqrt(lam)).cpu().numpy().astype(np.float64)
w = (Xt - mt) @ Wm_np
oor0 = None
for it in range(120):
    r_cur = np.linalg.norm(w, axis=1)
    fac = (r_g / r_cur) ** 0.5 if it < 100 else (r_g / r_cur)
    Xt += (((fac - 1.0)[:, None] * w) * ln) @ Vn.T
    if it == 0:
        oor0 = float(((Xt < 0) | (Xt > 255)).mean())
    np.clip(Xt, 0.0, 255.0, out=Xt)
    w = (Xt - mt) @ Wm_np
r_fin = np.linalg.norm(w, axis=1)
sev_err = np.abs(r_fin - r_g)
print(f"transport+projection: raw update leaves box at {oor0:.1%}; final "
      f"|sev-target| med={np.median(sev_err):.4f} p95={np.quantile(sev_err,.95):.4f} "
      f"max={sev_err.max():.3f}; within 0.05: {(sev_err < 0.05).mean():.1%}", flush=True)

he3 = hold_idx[Rh >= tau3]
Zh3 = Wall[he3] / R[he3][:, None]
mB = Rf >= np.quantile(Rf, 0.98)
Zdir = Wall[fit_idx][mB] / Rf[mB][:, None]
mZ, sZ = Zdir.mean(0), Zdir.std(0)
pca2 = np.linalg.svd((Zdir - mZ) / sZ, full_matrices=False)[2][:2]
Zg = w / np.linalg.norm(w, axis=1, keepdims=True)
Ph = ((Zh3 - mZ) / sZ) @ pca2.T
Pg = ((Zg - mZ) / sZ) @ pca2.T
print(f"shape validation (fit-tail PC basis, n_held={len(he3)}):", flush=True)
for k_, nm in enumerate(["PC1", "PC2"]):
    print(f"  {nm}: gen mean/sd={Pg[:,k_].mean():+.2f}/{Pg[:,k_].std():.2f}  "
          f"held-out={Ph[:,k_].mean():+.2f}/{Ph[:,k_].std():.2f}", flush=True)
he = hold_idx[Rh >= tau]
Rh_e = R[he]
print(f"severity | >= tau: gen q50/90={np.quantile(r_fin,[.5,.9]).round(2)} "
      f"held-out={np.quantile(Rh_e,[.5,.9]).round(2)}  (n_held={len(he)})", flush=True)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8.5, "text.color": INK})
NSHOW = 6
sev_targets = np.quantile(Rh_e, np.linspace(0.10, 0.95, NSHOW))
pairs = []
for s_ in sev_targets:
    ih = int(np.argmin(np.abs(Rh_e - s_)))
    ig = int(np.argmin(np.abs(r_fin - Rh_e[ih])))
    pairs.append((ih, ig))
print("figure pairs (real, gen):",
      [(round(Rh_e[a1], 2), round(float(r_fin[b1]), 2)) for a1, b1 in pairs], flush=True)
fig, axs = plt.subplots(2, NSHOW, figsize=(11.0, 4.1), facecolor=SURFACE)
for jj, (ih, ig) in enumerate(pairs):
    axs[0, jj].imshow(X8[he[ih]])
    axs[0, jj].set_title(f"real  R={Rh_e[ih]:.1f}", fontsize=8, pad=2)
    axs[1, jj].imshow(np.clip(Xt[ig], 0, 255).reshape(32, 32, 3).astype(np.uint8))
    axs[1, jj].set_title(f"generated  R={r_fin[ig]:.1f}", fontsize=8, pad=2)
    for row in range(2):
        axs[row, jj].set_xticks([]); axs[row, jj].set_yticks([])
fig.suptitle("EDM (NVIDIA) shapes carried to GPD-clock severities beyond the "
             "1-in-10,000 level: real held-out images (top), transported EDM "
             "samples (bottom), at matched severities",
             fontsize=10, fontweight="bold")
fig.savefig(os.path.expandvars("${PROJECT_ROOT}/docs/audit_edm.png"),
            dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote audit_edm.png   ({time.time()-T0:.0f}s)")
print("DONE", flush=True)
