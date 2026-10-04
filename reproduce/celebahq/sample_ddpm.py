"""Sample DDPM (CelebA-HQ-256, DDIM-100) into the 32x32 audit representation."""
import os
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_CACHE}"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

T0 = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"
P3 = os.path.expandvars("${CACHE_ROOT}/paper")
CACHE = os.environ.get("SMOKE_CACHE", f"{P3}/d2_ddpm_samples_cache.npz")
TARGET_FILE = CACHE.replace("_samples_cache.npz", "_target.txt")
SEED_BASE = 72000
NSTEPS = 100
M_FULL, M_SLOW = 30_000, 20_000
BS = 2 if SMOKE else 64
DEVICE = torch.device("cpu") if SMOKE else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
FP16 = DEVICE.type == "cuda"
WALL_GUARD = 3.2 * 3600

from diffusers import UNet2DModel, DDIMScheduler  # noqa: E402

unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(DEVICE)
unet.eval()
if FP16:
    unet = unet.half()
sched = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
sched.set_timesteps(NSTEPS)
steps = sched.timesteps
assert len(steps) == NSTEPS, f"respacing failed: {len(steps)} steps"
print(f"loaded ddpm-celebahq-256 ({sum(p.numel() for p in unet.parameters())/1e6:.0f}M"
      f" params) fp16={FP16}; DDIM respaced to {len(steps)} steps "
      f"({time.time()-T0:.0f}s)", flush=True)
if SMOKE:
    steps = steps[:2]
    print("SMOKE: running only 2 of the 100 timesteps", flush=True)

# effective target: decided once from measured throughput, persisted
M_target = 4 if SMOKE else None
if not SMOKE and os.path.exists(TARGET_FILE):
    M_target = int(open(TARGET_FILE).read().strip())
    print(f"target from {TARGET_FILE}: M={M_target}", flush=True)

MMAX = 4 if SMOKE else M_FULL
done = 0
X32 = np.zeros((MMAX, 32, 32, 3), dtype=np.uint8)
if os.path.exists(CACHE):
    prev = np.load(CACHE)["X32"]
    done = min(len(prev), MMAX)
    X32[:done] = prev[:done]
    print(f"resumed cache: {done} samples", flush=True)

gt = torch.Generator(device=DEVICE)
gt.manual_seed(SEED_BASE + done)
print(f"seed base {SEED_BASE}, generator seeded at {SEED_BASE + done}",
      flush=True)

t_sample0, n_sampled_here = None, 0
M_now = min(M_target, MMAX) if M_target is not None else MMAX
while done < M_now and time.time() - T0 < WALL_GUARD:
    nb = min(BS, M_now - done)
    x = torch.randn((nb, 3, 256, 256), device=DEVICE, generator=gt)
    if t_sample0 is None:
        t_sample0 = time.time()
    with torch.no_grad():
        for t in steps:
            eps = unet(x.half() if FP16 else x, t).sample.float()
            x = sched.step(eps, t, x).prev_sample
    px256 = ((x / 2 + 0.5).clamp(0, 1) * 255).round()
    x32 = torch.nn.functional.avg_pool2d(px256, 8)
    hwc = x32.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
    X32[done:done+nb] = hwc.cpu().numpy()
    done += nb
    n_sampled_here += nb
    rate = n_sampled_here / max(time.time() - t_sample0, 1e-9) * 3600
    if M_target is None and n_sampled_here >= 2 * BS:
        M_target = M_FULL if rate >= 2500 else M_SLOW
        M_now = min(M_target, MMAX)
        with open(TARGET_FILE, "w") as f:
            f.write(str(M_target))
        print(f"THROUGHPUT={rate:.0f} imgs/h -> target M={M_target} "
              f"({'full 30k' if M_target == M_FULL else 'RESCALED to 20k, disclose'})",
              flush=True)
    if n_sampled_here % (4 * BS) < BS or done >= M_now:
        print(f"  {done}/{M_now} sampled  ({time.time()-T0:.0f}s, "
              f"{rate:.0f} imgs/h steady-state)", flush=True)
        np.savez_compressed(CACHE, X32=X32[:done])

np.savez_compressed(CACHE, X32=X32[:done])
if t_sample0 is not None and n_sampled_here > 0:
    print(f"IMGS_PER_H={n_sampled_here/max(time.time()-t_sample0,1e-9)*3600:.0f}",
          flush=True)
print(f"cache {CACHE}: {done} samples, "
      f"min={X32[:done].min()} max={X32[:done].max()}", flush=True)
print("DONE" if (M_target is not None and done >= M_target) else
      f"PARTIAL {done}/{M_target or M_FULL} (resume-safe)", flush=True)
