"""Sample iDDPM (ImageNet-64, DDIM-50) into the 32x32 audit representation."""
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import CKPT, EXT, M_OF                       # noqa: E402
from sampling import parse, Cache                        # noqa: E402

T0 = time.time()
args = parse(default_bs=128).parse_args()
sys.path.insert(0, f"{EXT}/improved-diffusion")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0 = 8 if args.smoke else M_OF["iDDPM"]
BS = 4 if args.smoke else args.bs
PT = f"{CKPT}/iddpm-imagenet64_cond_270M_250K.pt"
print(f"iDDPM sampling: M0={M0} BS={BS} device={DEVICE}", flush=True)

cache = Cache("iddpm", M0, args.smoke, T0)
if cache.done < M0:
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
    print(f"loaded iDDPM ({n_par/1e6:.0f}M params, DDIM-50)  ({time.time()-T0:.0f}s)", flush=True)
    cache.start()
    while cache.done < M0 and time.time() - T0 < args.wall_guard:
        nb = min(BS, M0 - cache.done)
        torch.manual_seed(5555 + cache.done)
        y = torch.randint(1000, (nb,), device=DEVICE)
        with torch.no_grad():
            x = diffusion.ddim_sample_loop(
                model, (nb, 3, 64, 64), clip_denoised=True,
                model_kwargs={"y": y}, device=DEVICE)
        px = ((x + 1) * 127.5).clamp(0, 255)                # (nb,3,64,64)
        px32 = torch.nn.functional.avg_pool2d(px.float(), 2)
        hwc = px32.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        cache.add(hwc.cpu().numpy())
        if cache.done % 6_400 < nb:
            cache.progress()
            cache.save()
    cache.finish(args.smoke, full_M=M_OF["iDDPM"])
else:
    print(f"cache already complete ({cache.done}); nothing to do", flush=True)
print("DONE", flush=True)
