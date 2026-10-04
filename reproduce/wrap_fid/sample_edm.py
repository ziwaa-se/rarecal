"""Sample EDM (ImageNet-64, Heun 18 steps) into the 32x32 audit representation."""
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
args = parse(default_bs=256).parse_args()
sys.path.insert(0, f"{EXT}/edm")                         # dnnlib, torch_utils for the pickle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0 = 8 if args.smoke else M_OF["EDM"]
BS = 4 if args.smoke else args.bs
PKL = f"{CKPT}/edm-imagenet-64x64-cond-adm.pkl"
print(f"EDM (Heun-18) sampling: M0={M0} BS={BS} device={DEVICE}", flush=True)


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


cache = Cache("edm", M0, args.smoke, T0)
if cache.done < M0:
    with open(PKL, "rb") as f:
        net = pickle.load(f)["ema"].to(DEVICE)
    net.eval()
    print(f"loaded EDM net (label_dim={net.label_dim}, res={net.img_resolution})  "
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
        px = (x * 127.5 + 128).clip(0, 255)                     # (nb,3,64,64)
        px32 = torch.nn.functional.avg_pool2d(px.float(), 2)    # box 2x2 -> 32
        hwc = px32.permute(0, 2, 3, 1).round().clip(0, 255).to(torch.uint8)
        cache.add(hwc.cpu().numpy())
        if cache.done % 12_800 < nb:
            cache.progress()
            cache.save()
    cache.finish(args.smoke, full_M=M_OF["EDM"])
else:
    print(f"cache already complete ({cache.done}); nothing to do", flush=True)
print("DONE", flush=True)
