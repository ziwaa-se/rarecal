"""Sample iDDPM (ImageNet-64, DDIM-50) and audit it under R1."""
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
sys.path.insert(0, os.path.expandvars("${PROJECT_ROOT}/external/improved-diffusion"))
PT = os.path.expandvars("${PROJECT_ROOT}/external/iddpm-imagenet64_cond_270M_250K.pt")
CACHE = os.path.expandvars("${PROJECT_ROOT}/docs/iddpm_samples_cache.npz")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
from audit import build_severity, audit_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0, BS = 100_000, 128
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
    from improved_diffusion.script_util import (
        create_model_and_diffusion, model_and_diffusion_defaults)
    cfg = model_and_diffusion_defaults()
    cfg.update(image_size=64, num_channels=192, num_res_blocks=3,
               learn_sigma=True, class_cond=True, diffusion_steps=4000,
               noise_schedule="cosine", rescale_learned_sigmas=False,
               rescale_timesteps=False, timestep_respacing="ddim50")
    model, diffusion = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(PT, map_location="cpu"))
    model.to(DEVICE).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded iDDPM ({n_par/1e6:.0f}M params, DDIM-50)  "
          f"({time.time()-T0:.0f}s)", flush=True)
    while done < M0 and time.time() - T0 < WALL_GUARD:
        nb = min(BS, M0 - done)
        torch.manual_seed(5555 + done)
        y = torch.randint(1000, (nb,), device=DEVICE)
        with torch.no_grad():
            x = diffusion.ddim_sample_loop(
                model, (nb, 3, 64, 64), clip_denoised=True,
                model_kwargs={"y": y}, device=DEVICE)
        px = ((x + 1) * 127.5).clamp(0, 255)                # (nb,3,64,64)
        px32 = torch.nn.functional.avg_pool2d(px.float(), 2)
        hwc = px32.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        X32[done:done+nb] = hwc.cpu().numpy()
        done += nb
        if done % 6_400 < BS:
            print(f"  {done}/{M0} sampled ({done/(time.time()-T0+1e-9)*3600:.0f} "
                  f"img/h)  ({time.time()-T0:.0f}s)", flush=True)
            np.savez_compressed(CACHE, X32=X32[:done])
    np.savez_compressed(CACHE, X32=X32[:done])
    print(f"sampling finished/paused at {done}  ({time.time()-T0:.0f}s)", flush=True)

audit_model("iDDPM (2021)", X32[:done], sev, DEVICE,
            os.path.expandvars("${PROJECT_ROOT}/docs/audit_iddpm.png"), t0=T0)
print("DONE", flush=True)
