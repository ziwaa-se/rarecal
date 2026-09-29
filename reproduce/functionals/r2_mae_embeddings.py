"""Compute MAE ViT-B/16 embeddings (mean of patch tokens) for real images and model samples, used by R2."""
import os
import sys
import glob
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tailaudit"))
import severity as sc

T0 = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("S2_SMOKE", "0") == "1"
WALL_GUARD = 3.2 * 3600
BS = 256 if not SMOKE else 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

import timm
from safetensors.torch import load_file

blob = glob.glob(os.path.expandvars("${HF_CACHE}/hub/"
                 "models--timm--vit_base_patch16_224.mae/snapshots/*/"
                 "model.safetensors"))[0]
model = timm.create_model("vit_base_patch16_224", num_classes=0)
sd = load_file(blob)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"MAE load: {len(sd)} tensors; missing={missing} unexpected={unexpected}",
      flush=True)
assert not unexpected, "unexpected checkpoint keys"
assert all(k.startswith("head.") for k in missing), f"backbone keys missing: {missing}"
model.eval().to(DEV)
if DEV.type == "cuda":
    model.half()
    MEAN, STD = MEAN.half(), STD.half()
MEAN, STD = MEAN.to(DEV), STD.to(DEV)


@torch.no_grad()
def embed(X_uint8_hwc):
    """(B,32,32,3) uint8 -> (B,768) float32 numpy."""
    x = torch.from_numpy(np.ascontiguousarray(X_uint8_hwc)).to(DEV)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    if DEV.type == "cuda":
        x = x.half()
    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear",
                                        antialias=True)
    x = (x - MEAN) / STD
    tok = model.forward_features(x)          # (B,197,768), post final norm
    return tok[:, 1:].mean(1).float().cpu().numpy()


def embed_array(X, out_path, tag, resume=False):
    n = len(X)
    E = np.zeros((n, 768), dtype=np.float16)
    done = 0
    prog = out_path + ".progress.npz"
    if resume and os.path.exists(prog):
        pp = np.load(prog)
        done = int(pp["done"])
        E[:done] = pp["E"][:done]
        print(f"  [{tag}] resumed at {done}", flush=True)
    while done < n:
        if time.time() - T0 > WALL_GUARD:
            np.savez(prog, E=E, done=done)
            print(f"  [{tag}] wall guard at {done}; progress saved", flush=True)
            sys.exit(0)
        nb = min(BS, n - done)
        E[done:done + nb] = embed(X[done:done + nb].reshape(nb, 32, 32, 3))
        done += nb
        if resume and done % 204_800 < BS and done < n:
            np.savez(prog, E=E, done=done)
        if done % 102_400 < BS:
            print(f"  [{tag}] {done}/{n} ({done/(time.time()-T0):.0f} img/s cum) "
                  f"({time.time()-T0:.0f}s)", flush=True)
    np.savez(out_path, E=E)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"  [{tag}] DONE {n} -> {out_path} ({time.time()-T0:.0f}s)", flush=True)


if SMOKE:
    X8 = np.load(sc.NPY, mmap_mode="r")
    e = embed(np.ascontiguousarray(X8[:8]))
    print(f"smoke: emb shape={e.shape} norm={np.linalg.norm(e, axis=1).round(2)}",
          flush=True)
    print("SMOKE OK", flush=True)
    sys.exit(0)

# reals first (the long pole), then model caches
real_out = f"{sc.P2}/mae_embed_real.npz"
if not os.path.exists(real_out):
    X8 = np.load(sc.NPY, mmap_mode="r")
    embed_array(X8, real_out, "real", resume=True)

for name, path in sc.CONFIGS:
    out = f"{sc.P2}/mae_embed_{sc.SLUG[name]}.npz"
    if os.path.exists(out):
        print(f"  [{name}] already embedded", flush=True)
        continue
    X = sc.load_cache(path)
    embed_array(X, out, name)
    del X

print("ALL EMBEDDINGS DONE", flush=True)
