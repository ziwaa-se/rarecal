"""Compute MAE ViT-B/16 embeddings for the M=50,000 DiT-XL/2 sample sets."""
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
WALL_GUARD = 0.6 * 3600          # stop before the time limit and resume later
BS = 256 if not SMOKE else 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

FINAL = [("dit_cfg15", f"{sc.TFP}/dit_cfg1p5_samples_cache.npz"),
         ("dit_cfg10", f"{sc.TFP}/dit_cfg1_samples_cache.npz")]

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


def embed_array(X, out_path, tag):
    n = len(X)
    E = np.zeros((n, 768), dtype=np.float16)
    done = 0
    prog = out_path + ".progress.npz"
    if os.path.exists(prog):
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
        if done % 25_600 < BS:
            np.savez(prog, E=E, done=done)
            print(f"  [{tag}] {done}/{n} ({done/(time.time()-T0):.0f} img/s cum) "
                  f"({time.time()-T0:.0f}s)", flush=True)
    np.savez(out_path, E=E)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"  [{tag}] DONE {n} -> {out_path} ({time.time()-T0:.0f}s)", flush=True)


if SMOKE:
    X = sc.load_cache(FINAL[0][1])
    print(f"smoke: final cache {FINAL[0][0]} M={len(X)}", flush=True)
    e = embed(np.ascontiguousarray(X[:8]))
    print(f"smoke: emb shape={e.shape} norm={np.linalg.norm(e, axis=1).round(2)}",
          flush=True)
    print("SMOKE OK", flush=True)
    sys.exit(0)

PARTIAL = {"dit_cfg15": f"{sc.P2}/mae_embed_dit_cfg15.npz",
        "dit_cfg10": f"{sc.P2}/mae_embed_dit_cfg10.npz"}
for slug, path in FINAL:
    out = f"{sc.P2}/mae_embed_{slug}_final.npz"
    if os.path.exists(out):
        print(f"  [{slug}] already embedded", flush=True)
        continue
    X = sc.load_cache(path)
    print(f"[{slug}] final cache M={len(X)}", flush=True)
    embed_array(X, out, slug)
    # report (not assert) agreement with the partial embeddings
    try:
        Es = np.load(PARTIAL[slug])["E"]
        Ef = np.load(out)["E"]
        nn = min(len(Es), len(Ef))
        d = np.abs(Ef[:nn].astype(np.float32) - Es[:nn].astype(np.float32))
        print(f"  [{slug}] partial-prefix agreement over {nn}: "
              f"max|diff|={d.max():.4f} mean|diff|={d.mean():.6f}", flush=True)
    except Exception as ex:
        print(f"  [{slug}] partial comparison skipped: {ex}", flush=True)
    del X

print("ALL FINAL DIT EMBEDDINGS DONE", flush=True)
