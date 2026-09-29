"""Paths and constants of the wrapping/FID experiment, configured through environment variables."""
import os

CODE = os.path.expandvars("${WRAPFID_ROOT}")
DATA = os.path.expandvars("${WRAPFID_STORE}")

NPY = f"{DATA}/data/imagenet32_train.npy"          # our own copy, 1,281,167 x 32x32x3 uint8
CKPT = f"{DATA}/ckpt"
EXT = f"{DATA}/external"                           # git clones of the public model repos
HF = f"{DATA}/hf_cache"
CACHES = f"{DATA}/caches"
WRAPPED = f"{DATA}/caches/wrapped"
LAYER_DIR = f"{DATA}/layer"
LAYER = f"{LAYER_DIR}/layer_seed8000.npz"
FEATS = f"{DATA}/features"
RESULTS = f"{DATA}/results"

os.environ.setdefault("HF_HOME", HF)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# (display name, slug, M per the paper's Table 1). EDM-churn deliberately omitted.
CONFIGS = [
    ("iDDPM",       "iddpm",       100_000),
    ("EDM",         "edm",         100_000),
    ("EDM2-S",      "edm2_s",      100_000),
    ("EDM2-M",      "edm2_m",      100_000),
    ("StyleGAN-XL", "stylegan_xl", 100_000),
    ("DiT-cfg1.5",  "dit_cfg15",    50_000),
    ("DiT-cfg1.0",  "dit_cfg10",    50_000),
]
SLUG = {n: s for n, s, _ in CONFIGS}
M_OF = {n: m for n, _, m in CONFIGS}


def cache_path(slug, smoke=False):
    if smoke:
        return f"{CACHES}/smoke/{slug}_smoke.npz"
    return f"{CACHES}/{slug}_samples_cache.npz"


# Frozen severity-layer constants printed by the paper (App. repro, data/leaderboard_r1.csv).
PAPER_LAYER = dict(q=0.9925, u=12.061159687419158, xi=-0.05046346279934458,
                   sg=0.9736608413583477, zeta=0.0075,
                   tau3=13.934952565857854, tau4=16.129062355967164)
N_HOLD_PAPER = 1_081_167
