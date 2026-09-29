"""Sample EDM2-S or EDM2-M (ImageNet-64) into the 32x32 audit representation."""
import os
import sys
import time
import pickle
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from paths import CKPT, EXT, M_OF                       # noqa: E402
from sampling import parse, Cache                        # noqa: E402

T0 = time.time()
ap = parse(default_bs=256)
ap.add_argument("--size", choices=["s", "m"], required=True)
args = ap.parse_args()
sys.path.insert(0, f"{EXT}/edm2")                        # dnnlib, torch_utils for the pickle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAME = {"s": "EDM2-S", "m": "EDM2-M"}[args.size]
SLUG = {"s": "edm2_s", "m": "edm2_m"}[args.size]
PKL = {"s": f"{CKPT}/edm2-img64-s-1073741-0.075.pkl",
       "m": f"{CKPT}/edm2-img64-m-2147483-0.060.pkl"}[args.size]
M0 = 8 if args.smoke else M_OF[NAME]
BS = 4 if args.smoke else args.bs
print(f"{NAME} (Heun-32) sampling: M0={M0} BS={BS} device={DEVICE}", flush=True)


def edm_sampler(net, noise, labels=None,
                num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
                dtype=torch.float32):
    # verbatim from NVlabs/edm2 generate_images.py, guidance=1 (deterministic Heun)
    def denoise(x, t):
        return net(x, t, labels).to(dtype)
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat, t_hat = x_next, t_cur
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


cache = Cache(SLUG, M0, args.smoke, T0)
if cache.done < M0:
    with open(PKL, "rb") as f:
        net = pickle.load(f)["ema"].to(DEVICE)
    net.eval()
    print(f"loaded {NAME} net (label_dim={net.label_dim}, res={net.img_resolution})  "
          f"({time.time()-T0:.0f}s)", flush=True)
    gt = torch.Generator(device=DEVICE)
    gt.manual_seed(1234 + cache.done)
    eye = torch.eye(net.label_dim, device=DEVICE)
    cache.start()
    while cache.done < M0 and time.time() - T0 < args.wall_guard:
        nb = min(BS, M0 - cache.done)
        lat = torch.randn((nb, net.img_channels, net.img_resolution,
                           net.img_resolution), device=DEVICE, generator=gt)
        cls = eye[torch.randint(net.label_dim, (nb,), device=DEVICE, generator=gt)]
        with torch.no_grad():
            x = edm_sampler(net, lat, cls)
        px = (x * 127.5 + 128).clip(0, 255)                     # StandardRGBEncoder.decode
        px32 = torch.nn.functional.avg_pool2d(px.float(), 2)    # box 2x2 -> 32
        hwc = px32.permute(0, 2, 3, 1).round().clip(0, 255).to(torch.uint8)
        cache.add(hwc.cpu().numpy())
        if cache.done % 12_800 < nb:
            cache.progress()
            cache.save()
    cache.finish(args.smoke, full_M=M_OF[NAME])
else:
    print(f"cache already complete ({cache.done}); nothing to do", flush=True)
print("DONE", flush=True)
