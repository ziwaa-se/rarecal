"""Sample StyleGAN-XL (ImageNet-32) into the audit representation."""
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
args = parse(default_bs=256).parse_args()
sys.path.insert(0, f"{EXT}/stylegan-xl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M0 = 8 if args.smoke else M_OF["StyleGAN-XL"]
BS = 4 if args.smoke else args.bs
PKL = f"{CKPT}/sgxl-imagenet32.pkl"
print(f"StyleGAN-XL sampling: M0={M0} BS={BS} device={DEVICE}", flush=True)

# custom CUDA ops: try the JIT plugin once; on any failure pin that op to its
# reference implementation instead of crashing the job
from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu   # noqa: E402
for _mod in (bias_act, upfirdn2d, filtered_lrelu):
    def _make_safe(mod, orig):
        def safe_init():
            try:
                return orig()
            except Exception as e:
                print(f"{mod.__name__}: plugin failed ({type(e).__name__}); using ref impl",
                      flush=True)
                mod._init = lambda: False
                return False
        return safe_init
    _mod._init = _make_safe(_mod, _mod._init)

cache = Cache("stylegan_xl", M0, args.smoke, T0)
if cache.done < M0:
    import dnnlib
    import legacy
    with dnnlib.util.open_url(PKL) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE).eval()
    print(f"loaded StyleGAN-XL G_ema (z_dim={G.z_dim}, c_dim={G.c_dim}, "
          f"res={G.img_resolution})  ({time.time()-T0:.0f}s)", flush=True)
    gt = torch.Generator(device=DEVICE)
    gt.manual_seed(4321 + cache.done)
    eye = torch.eye(G.c_dim, device=DEVICE)
    cache.start()
    while cache.done < M0 and time.time() - T0 < args.wall_guard:
        nb = min(BS, M0 - cache.done)
        z = torch.randn((nb, G.z_dim), device=DEVICE, generator=gt)
        cls = eye[torch.randint(G.c_dim, (nb,), device=DEVICE, generator=gt)]
        with torch.no_grad():
            img = G(z, cls, truncation_psi=1.0, noise_mode="const")
        px = (img * 127.5 + 128).clip(0, 255)               # NCHW, native 32x32
        hwc = px.permute(0, 2, 3, 1).round().clip(0, 255).to(torch.uint8)
        cache.add(hwc.cpu().numpy())
        if cache.done % 25_600 < nb:
            cache.progress()
            cache.save()
    cache.finish(args.smoke, full_M=M_OF["StyleGAN-XL"])
else:
    print(f"cache already complete ({cache.done}); nothing to do", flush=True)
print("DONE", flush=True)
