"""Feature extractors (Inception-v3, MAE ViT-B/16) and FID, KID and FMD."""
import glob
import time
import numpy as np
import torch

from paths import HF

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ Inception
def inception_extractor(device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    fe = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"])
    return fe.to(device).eval()


@torch.no_grad()
def inception_features(X_uint8_hwc, fe, device, bs=500, tag="", t0=None):
    n = len(X_uint8_hwc)
    F = np.empty((n, 2048), dtype=np.float32)
    for a in range(0, n, bs):
        x = torch.from_numpy(np.array(X_uint8_hwc[a:a + bs].reshape(-1, 32, 32, 3)))
        x = x.permute(0, 3, 1, 2).contiguous().to(device)      # uint8 NCHW
        F[a:a + bs] = fe(x)[0].float().cpu().numpy()
        if tag and t0 is not None and (a // bs) % 200 == 0:
            print(f"    [inception {tag}] {a}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return F


# ------------------------------------------------------------------ MAE
def mae_model(device):
    import timm
    from safetensors.torch import load_file
    blob = glob.glob(f"{HF}/hub/models--timm--vit_base_patch16_224.mae/snapshots/*/model.safetensors")[0]
    model = timm.create_model("vit_base_patch16_224", num_classes=0)
    sd = load_file(blob)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected MAE keys: {unexpected}"
    assert all(k.startswith("head.") for k in missing), f"backbone keys missing: {missing}"
    model = model.eval().to(device)
    if torch.device(device).type == "cuda":
        model = model.half()
    return model


@torch.no_grad()
def mae_features(X_uint8_hwc, model, device, bs=256, tag="", t0=None):
    dev = torch.device(device)
    half = dev.type == "cuda"
    mean = (MEAN.half() if half else MEAN).to(dev)
    std = (STD.half() if half else STD).to(dev)
    n = len(X_uint8_hwc)
    E = np.empty((n, 768), dtype=np.float32)
    for a in range(0, n, bs):
        x = torch.from_numpy(np.array(X_uint8_hwc[a:a + bs].reshape(-1, 32, 32, 3))).to(dev)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if half:
            x = x.half()
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", antialias=True)
        x = (x - mean) / std
        tok = model.forward_features(x)
        E[a:a + bs] = tok[:, 1:].mean(1).float().cpu().numpy()
        if tag and t0 is not None and (a // bs) % 400 == 0:
            print(f"    [mae {tag}] {a}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return E


# ------------------------------------------------------------------ metrics
def stats(F, chunk=50_000):
    """(mu, Sigma) with Sigma the ddof=1 sample covariance, float64, chunked."""
    n, k = F.shape
    mu = np.zeros(k, np.float64)
    S = np.zeros((k, k), np.float64)
    for a in range(0, n, chunk):
        xb = np.asarray(F[a:a + chunk], dtype=np.float64)
        mu += xb.sum(0)
        S += xb.T @ xb
    mu /= n
    C = (S - n * np.outer(mu, mu)) / (n - 1)
    return mu, C


def frechet(mu1, C1, mu2, C2):
    """||mu1-mu2||^2 + tr + tr 2 tr (^{1/2} ^{1/2})^{1/2}, via eigh."""
    d = np.asarray(mu1, np.float64) - np.asarray(mu2, np.float64)
    C1 = np.asarray(C1, np.float64)
    C2 = np.asarray(C2, np.float64)
    ev, U = np.linalg.eigh((C1 + C1.T) / 2)
    A = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
    Mx = A @ ((C2 + C2.T) / 2) @ A
    ev2 = np.linalg.eigvalsh((Mx + Mx.T) / 2)
    return float(d @ d + np.trace(C1) + np.trace(C2) - 2.0 * np.sqrt(np.clip(ev2, 0, None)).sum())


def kid(F1, F2, seed=2020):
    """KID mean/std, torch-fidelity defaults (poly-3, 100 subsets x 1000)."""
    from torch_fidelity.metric_kid import kid_features_to_metric
    out = kid_features_to_metric(torch.from_numpy(np.ascontiguousarray(F1, dtype=np.float32)),
                                 torch.from_numpy(np.ascontiguousarray(F2, dtype=np.float32)),
                                 verbose=False, rng_seed=seed)
    return out["kernel_inception_distance_mean"], out["kernel_inception_distance_std"]
