"""Per-image quality measures of transported samples: PSNR, SSIM, LPIPS, saturation and improved precision."""
import numpy as np
import torch


# ------------------------------------------------------------------ per-image change
def to_nchw(X_uint8_hwc, device):
    x = torch.from_numpy(np.array(X_uint8_hwc.reshape(-1, 32, 32, 3))).to(device)
    return x.permute(0, 3, 1, 2).float()                       # 0..255


def lpips_net(device):
    from torchmetrics.functional.image.lpips import _NoTrainLpips
    return _NoTrainLpips(net="alex").to(device).eval()


@torch.no_grad()
def pair_metrics(A_uint8, B_uint8, device, net=None, bs=250):
    """Per-image metrics between image sets A (before) and B (after), same order."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    n = len(A_uint8)
    out = {k: np.empty(n, np.float64) for k in ("l2", "mse", "mean_abs", "ssim", "lpips")}
    for a in range(0, n, bs):
        xa, xb = to_nchw(A_uint8[a:a + bs], device), to_nchw(B_uint8[a:a + bs], device)
        d = (xb - xa).flatten(1)
        out["l2"][a:a + bs] = d.norm(dim=1).cpu().numpy()
        out["mse"][a:a + bs] = (d ** 2).mean(1).cpu().numpy()
        out["mean_abs"][a:a + bs] = d.abs().mean(1).cpu().numpy()
        out["ssim"][a:a + bs] = ssim_fn(xb, xa, data_range=255.0, reduction="none").cpu().numpy()
        if net is not None:
            ua = torch.nn.functional.interpolate(xa / 255.0, size=224, mode="bilinear", antialias=True)
            ub = torch.nn.functional.interpolate(xb / 255.0, size=224, mode="bilinear", antialias=True)
            out["lpips"][a:a + bs] = net(ua.clamp(0, 1), ub.clamp(0, 1), normalize=True).flatten().cpu().numpy()
        else:
            out["lpips"][a:a + bs] = np.nan
    with np.errstate(divide="ignore"):
        out["psnr"] = 10.0 * np.log10(255.0 ** 2 / out["mse"])  # inf where the image did not change
    return out


def saturation(X_uint8):
    """Fraction of pixel values sitting on the box boundary (0 or 255), per image."""
    Xf = X_uint8.reshape(len(X_uint8), -1)
    return ((Xf == 0) | (Xf == 255)).mean(1)


# ------------------------------------------------------------------ realism (improved precision)
@torch.no_grad()
def knn_radii(F, k=3, chunk=4000):
    """Distance from each reference row to its k-th nearest OTHER reference row."""
    r = torch.empty(len(F), device=F.device, dtype=F.dtype)
    for a in range(0, len(F), chunk):
        d = torch.cdist(F[a:a + chunk], F)
        r[a:a + chunk] = d.kthvalue(k + 1, dim=1).values       # +1 skips the zero self-distance
    return r


@torch.no_grad()
def realism(Fq, F_man, radii, sub_mask=None, chunk=4000):
    """For query features Fq: (inside manifold?, NN distance, NN index within sub_mask rows)."""
    inside = torch.empty(len(Fq), dtype=torch.bool, device=Fq.device)
    nn_d = torch.empty(len(Fq), device=Fq.device, dtype=Fq.dtype)
    nn_sub = torch.empty(len(Fq), dtype=torch.long, device=Fq.device)
    sub = torch.arange(len(F_man), device=Fq.device) if sub_mask is None else torch.where(sub_mask)[0]
    for a in range(0, len(Fq), chunk):
        d = torch.cdist(Fq[a:a + chunk], F_man)
        inside[a:a + chunk] = (d <= radii[None, :]).any(1)
        nn_d[a:a + chunk] = d.min(1).values
        nn_sub[a:a + chunk] = d[:, sub].argmin(1)
    return inside.cpu().numpy(), nn_d.cpu().numpy(), nn_sub.cpu().numpy()


def kid_small(F1, F2, subset, seed=2020):
    from torch_fidelity.metric_kid import kid_features_to_metric
    out = kid_features_to_metric(torch.from_numpy(np.ascontiguousarray(F1, dtype=np.float32)),
                                 torch.from_numpy(np.ascontiguousarray(F2, dtype=np.float32)),
                                 verbose=False, rng_seed=seed, kid_subset_size=int(subset))
    return out["kernel_inception_distance_mean"], out["kernel_inception_distance_std"]


# ------------------------------------------------------------------ paired bootstrap FID
class FrechetRef:
    """Fixed real reference (mu, C); evaluates FID of a feature batch in float64 on `device`."""

    def __init__(self, mu_ref, C_ref, device):
        self.dev = torch.device(device)
        self.mu = torch.tensor(mu_ref, dtype=torch.float64, device=self.dev)
        C = torch.tensor(C_ref, dtype=torch.float64, device=self.dev)
        ev, U = torch.linalg.eigh((C + C.T) / 2)
        self.A = (U * ev.clamp_min(0).sqrt()) @ U.T             # C_ref^{1/2}
        self.trC = torch.trace(C)

    @torch.no_grad()
    def fid(self, Fb):
        Fb = Fb.to(self.dev, torch.float64)
        n = len(Fb)
        mu = Fb.mean(0)
        Xc = Fb - mu
        C = (Xc.T @ Xc) / (n - 1)
        Mx = self.A @ C @ self.A
        ev = torch.linalg.eigvalsh((Mx + Mx.T) / 2)
        d = mu - self.mu
        return float(d @ d + torch.trace(C) + self.trC - 2.0 * ev.clamp_min(0).sqrt().sum())


# ------------------------------------------------------------------ gallery
def gallery(path_base, title, rows, col_titles):
    """rows: list of (row label, uint8 array (ncol,32,32,3)). Saves.png and.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(col_titles)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(1.35 * ncol + 0.9, 1.45 * len(rows) + 0.6),
                            squeeze=False)
    for i, (lab, imgs) in enumerate(rows):
        for j in range(ncol):
            ax = axs[i, j]
            ax.imshow(imgs[j], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=6.5, pad=2)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
    fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path_base + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
