"""Sample StyleGAN2-ADA (CelebA-HQ-256) into the 32x32 audit representation."""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"
sys.path.insert(0, os.path.expandvars("${CACHE_ROOT}/external/stylegan-xl"))
PKL = (os.path.expandvars("${CACHE_ROOT}/external/"
       "celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl"))
CACHE = os.environ.get(
    "SMOKE_CACHE",
    os.path.expandvars("${CACHE_ROOT}/paper/d2_sg2ada_samples_cache.npz"))
SEED_BASE = 71000
M0, BS = (4, 2) if SMOKE else (100_000, 128)
DEVICE = torch.device("cpu") if SMOKE else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
WALL_GUARD = 3.2 * 3600

# custom CUDA ops: try the JIT plugin once; on any failure pin that op to its
# reference implementation instead of crashing the job (audit_stylegan_xl.py pattern)
from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu  # noqa: E402
for _mod in (bias_act, upfirdn2d, filtered_lrelu):
    def _make_safe(mod, orig):
        def safe_init():
            try:
                return orig()
            except Exception as e:
                print(f"{mod.__name__}: plugin failed ({e}); using ref impl",
                      flush=True)
                mod._init = lambda: False
                return False
        return safe_init
    _mod._init = _make_safe(_mod, _mod._init)

done = 0
X32 = np.zeros((M0, 32, 32, 3), dtype=np.uint8)
if os.path.exists(CACHE):
    prev = np.load(CACHE)["X32"]
    done = min(len(prev), M0)
    X32[:done] = prev[:done]
    print(f"resumed cache: {done} samples", flush=True)

if done < M0:
    import dnnlib
    import legacy
    with dnnlib.util.open_url(PKL) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE).eval()
    print(f"loaded SG2-ADA G_ema (z_dim={G.z_dim}, c_dim={G.c_dim}, "
          f"res={G.img_resolution})  ({time.time()-T0:.0f}s)", flush=True)
    gt = torch.Generator(device=DEVICE)
    gt.manual_seed(SEED_BASE + done)
    print(f"seed base {SEED_BASE}, generator seeded at {SEED_BASE + done}",
          flush=True)
    t_first = None
    while done < M0 and time.time() - T0 < WALL_GUARD:
        nb = min(BS, M0 - done)
        z = torch.randn((nb, G.z_dim), device=DEVICE, generator=gt)
        with torch.no_grad():
            img = G(z, None, truncation_psi=1.0, noise_mode="const",
                    **({"force_fp32": True} if SMOKE else {}))
        # uint8 quantization at 256^2 first (corpus identity), then 8x box mean
        px256 = (img.float() * 127.5 + 128).clamp(0, 255).round()
        x32 = F.avg_pool2d(px256, 8)
        hwc = x32.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        X32[done:done+nb] = hwc.cpu().numpy()
        done += nb
        if t_first is None:
            t_first = time.time()
            print(f"first batch done ({t_first-T0:.0f}s)", flush=True)
        if done % 12_800 < BS:
            rate = (done) / max(time.time() - T0, 1) * 3600
            print(f"  {done}/{M0} sampled  ({time.time()-T0:.0f}s, "
                  f"~{rate:.0f} imgs/h incl. startup)", flush=True)
            np.savez_compressed(CACHE, X32=X32[:done])
    np.savez_compressed(CACHE, X32=X32[:done])
    print(f"sampling finished at {done}  ({time.time()-T0:.0f}s)", flush=True)

print(f"cache {CACHE}: {done} samples, "
      f"min={X32[:done].min()} max={X32[:done].max()}", flush=True)
print("DONE" if done >= M0 else f"PARTIAL {done}/{M0} (resume-safe)",
      flush=True)
