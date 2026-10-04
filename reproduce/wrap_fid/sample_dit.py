"""Sample DiT-XL/2 (ImageNet-256, DDIM-50) into the 32x32 audit representation."""
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import CKPT, EXT, M_OF, HF                   # noqa: E402 (sets HF_HOME, offline)
from sampling import parse, Cache                        # noqa: E402

T0 = time.time()
ap = parse(default_bs=96)
ap.add_argument("--cfg", type=float, required=True)
args = ap.parse_args()
sys.path.insert(0, f"{EXT}/DiT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = args.cfg
NAME = f"DiT-cfg{CFG:.1f}"
SLUG = f"dit_cfg{CFG:.1f}".replace(".", "")
assert NAME in M_OF, NAME
M0 = 4 if args.smoke else M_OF[NAME]
BS = 2 if args.smoke else args.bs
PT = f"{CKPT}/DiT-XL-2-256x256.pt"
print(f"{NAME} sampling: M0={M0} BS={BS} device={DEVICE} slug={SLUG}", flush=True)

cache = Cache(SLUG, M0, args.smoke, T0)
if cache.done < M0:
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
    print(f"loaded DiT-XL/2 ({n_par/1e6:.0f}M) + sd-vae-ft-ema, DDIM-50, cfg={CFG}  "
          f"({time.time()-T0:.0f}s)", flush=True)
    cache.start()
    while cache.done < M0 and time.time() - T0 < args.wall_guard:
        nb = min(BS, M0 - cache.done)
        torch.manual_seed(7777 + cache.done)
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
        cache.add(hwc.cpu().numpy())
        if cache.done % 4_800 < nb:
            cache.progress()
            cache.save()
    cache.finish(args.smoke, full_M=M_OF[NAME])
else:
    print(f"cache already complete ({cache.done}); nothing to do", flush=True)
print("DONE", flush=True)
