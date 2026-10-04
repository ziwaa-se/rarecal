"""Sample DiT-XL/2 (ImageNet-256, guidance 1.5 or 1.0) and audit it under R1."""
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
sys.path.insert(0, os.path.expandvars("${PROJECT_ROOT}/external/DiT"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
from audit import build_severity, audit_model

CFG = float(os.environ.get("DIT_CFG", "1.5"))
TAG = f"cfg{CFG:g}".replace(".", "p")
PT = os.path.expandvars("${PROJECT_ROOT}/external/DiT-XL-2-256x256.pt")
CACHE = os.path.expandvars(f"${CACHE_ROOT}/dit_{TAG}_samples_cache.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0, BS = 50_000, 96
WALL_GUARD = 3.2 * 3600
print(f"DiT-XL/2 audit, cfg={CFG}, cache={os.path.basename(CACHE)}", flush=True)

sev = build_severity(DEVICE, T0)

done = 0
X32 = np.zeros((M0, 32, 32, 3), dtype=np.uint8)
if os.path.exists(CACHE):
    prev = np.load(CACHE)["X32"]
    done = min(len(prev), M0)
    X32[:done] = prev[:done]
    print(f"resumed cache: {done} samples", flush=True)

if done < M0:
    from models import DiT_models
    from diffusion import create_diffusion
    from diffusers.models import AutoencoderKL
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(DEVICE)
    model.load_state_dict(torch.load(PT, map_location="cpu"))
    model.eval()
    diffusion = create_diffusion("ddim50")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)
    vae.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded DiT-XL/2 ({n_par/1e6:.0f}M) + sd-vae-ft-ema, DDIM-50  "
          f"({time.time()-T0:.0f}s)", flush=True)
    while done < M0 and time.time() - T0 < WALL_GUARD:
        nb = min(BS, M0 - done)
        torch.manual_seed(7777 + done)
        z = torch.randn(nb, 4, 32, 32, device=DEVICE)
        y = torch.randint(1000, (nb,), device=DEVICE)
        with torch.no_grad():
            if CFG > 1.0:
                z2 = torch.cat([z, z], 0)
                y2 = torch.cat([y, torch.full_like(y, 1000)], 0)
                kw = dict(y=y2, cfg_scale=CFG)
                lat = diffusion.ddim_sample_loop(
                    model.forward_with_cfg, z2.shape, z2, clip_denoised=False,
                    model_kwargs=kw, progress=False, device=DEVICE)
                lat, _ = lat.chunk(2, dim=0)
            else:
                lat = diffusion.ddim_sample_loop(
                    model, z.shape, z, clip_denoised=False,
                    model_kwargs=dict(y=y), progress=False, device=DEVICE)
            x = vae.decode(lat / 0.18215).sample          # (nb,3,256,256) in [-1,1]
        px = (x * 127.5 + 128).clamp(0, 255)
        px32 = torch.nn.functional.avg_pool2d(px.float(), 8)
        hwc = px32.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        X32[done:done+nb] = hwc.cpu().numpy()
        done += nb
        if done % 4_800 < BS:
            print(f"  {done}/{M0} sampled ({done/(time.time()-T0+1e-9)*3600:.0f} "
                  f"img/h)  ({time.time()-T0:.0f}s)", flush=True)
            np.savez_compressed(CACHE, X32=X32[:done])
    np.savez_compressed(CACHE, X32=X32[:done])
    print(f"sampling finished/paused at {done}  ({time.time()-T0:.0f}s)", flush=True)

audit_model(f"DiT-XL/2 cfg={CFG:g}", X32[:done], sev, DEVICE,
            os.path.expandvars(f"${PROJECT_ROOT}/docs/dit_{TAG}_audit.png"),
            nkeep=500, t0=T0)
print("DONE", flush=True)
