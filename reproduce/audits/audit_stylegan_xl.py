"""Sample StyleGAN-XL (ImageNet-32) and audit it under R1."""
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
# deps (timm 0.6.13, ftfy, ninja) come from PYTHONPATH
sys.path.insert(0, os.path.expandvars("${PROJECT_ROOT}/external/stylegan-xl"))
PKL = os.path.expandvars("${PROJECT_ROOT}/external/sgxl-imagenet32.pkl")
CACHE = os.path.expandvars("${PROJECT_ROOT}/docs/sgxl_samples_cache.npz")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
from audit import build_severity, audit_model, D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0, BS = 100_000, 256

# custom CUDA ops: try the JIT plugin once; on any failure pin that op to its
# reference implementation instead of crashing the job
from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu
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
WALL_GUARD = 3.2 * 3600

sev = build_severity(DEVICE, T0)

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
    print(f"loaded StyleGAN-XL G_ema (z_dim={G.z_dim}, c_dim={G.c_dim}, "
          f"res={G.img_resolution})  ({time.time()-T0:.0f}s)", flush=True)
    gt = torch.Generator(device=DEVICE)
    gt.manual_seed(4321 + done)
    eye = torch.eye(G.c_dim, device=DEVICE)
    while done < M0 and time.time() - T0 < WALL_GUARD:
        nb = min(BS, M0 - done)
        z = torch.randn((nb, G.z_dim), device=DEVICE, generator=gt)
        cls = eye[torch.randint(G.c_dim, (nb,), device=DEVICE, generator=gt)]
        with torch.no_grad():
            img = G(z, cls, truncation_psi=1.0, noise_mode="const")
        px = (img * 127.5 + 128).clip(0, 255)               # NCHW, native 32x32
        hwc = px.permute(0, 2, 3, 1).round().clip(0, 255).to(torch.uint8)
        X32[done:done+nb] = hwc.cpu().numpy()
        done += nb
        if done % 25_600 < BS:
            print(f"  {done}/{M0} sampled  ({time.time()-T0:.0f}s)", flush=True)
            np.savez_compressed(CACHE, X32=X32[:done])
    np.savez_compressed(CACHE, X32=X32[:done])
    print(f"sampling finished at {done}  ({time.time()-T0:.0f}s)", flush=True)

audit_model("StyleGAN-XL", X32[:done], sev, DEVICE,
            os.path.expandvars("${PROJECT_ROOT}/docs/audit_stylegan_xl.png"), t0=T0)
print("DONE", flush=True)
