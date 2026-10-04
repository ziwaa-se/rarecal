# Reproducing the real-model study

This folder holds the full pipeline behind the numbers in `results/`: sampling from public
checkpoints, the three severity functionals, the audits, the validation studies, the wrapper
and its effect on FID. It uses its own implementation (`tailaudit/`) of the same estimators
that the `rarecal` package exposes.

Most readers only need **level 2**: every figure and table can be redrawn from the CSV files
without a GPU.

```bash
python reproduce/figures/make_figures.py          # main figures, from ../results
python reproduce/figures/fig_quality_axis.py      # FID / FMD versus tail ratio
python reproduce/figures/fig_celebahq.py          # CelebA-HQ replication
python reproduce/wrap_fid/make_tables_and_figure.py   # wrapping and FID: tables + summary figure
```

Outputs are written next to the scripts (set `TAILAUDIT_FIGS` to change that).

## Layout

| Folder | Content |
|---|---|
| `tailaudit/` | shared code: severity functional R1, calibration law (spliced generalized Pareto), Wilson and quadrature intervals, verdicts (`severity.py`); per-model audit and transport (`audit.py`); features, Fréchet distances and mixtures (`analysis_common.py`) |
| `audits/` | sample each public checkpoint and audit it: iDDPM, EDM, EDM (stochastic), EDM2-S/M, StyleGAN-XL, DiT-XL/2 |
| `functionals/` | calibration law for R1, the embedding functional R2 (MAE ViT-B/16), the worst-patch functional R3, agreement between functionals |
| `validation/` | real-versus-real splits and re-splits, planted ratios and coverage, sensitivity, interval width by budget, ablations, choice of the generalized-Pareto threshold |
| `experiments/` | moment-matched mixtures on real features and the kernel contrast, FMD quality axis, tail shape (dispersion) test, wrapper on every model, downstream estimates, result tables |
| `celebahq/` | CelebA-HQ 256 replication: calibration law, StyleGAN2-ADA and DDPM sampling, audit and wrapping |
| `wrap_fid/` | effect of wrapping on FID/KID/FMD: sampling, wrapping, paired bootstrap, per-image quality, optional guard, tables and figure |
| `figures/` | figure scripts that read only `../results` |

## Data and checkpoints

**Reference data.** ImageNet-32 training set (1,281,167 images, `uint8`, 32×32×3), split once
with seed 8000 into 200,000 calibration images (PCA chart, calibration law, shape basis) and
1,081,167 held-out images (all audited thresholds). CelebA-HQ 256 (30,000 images),
box-downsampled by 8 to 32×32, split with the same seed into 10,000 / 20,000.

**Common grid.** Every sample is quantized to `uint8` at its native resolution, box-averaged
to 32×32 and rounded before the severity is computed; the real data follow the same path.

**Checkpoints** (public releases, unchanged, with the sampler settings of their repositories):

| Model | Checkpoint | Sampler |
|---|---|---|
| iDDPM | `imagenet64_cond_270M_250K.pt` | DDIM-50, cosine schedule, learned sigma |
| EDM | `edm-imagenet-64x64-cond-adm.pkl` | Heun, 18 steps (NFE 35) |
| EDM (stochastic) | same | 256 steps, S_churn 40, S_min 0.05, S_max 50, S_noise 1.003 |
| EDM2-S / EDM2-M | `edm2-img64-s-1073741-0.075.pkl` / `edm2-img64-m-2147483-0.060.pkl` | Heun, 32 steps (NFE 63) |
| StyleGAN-XL | `imagenet32.pkl` (G_ema) | one pass, no truncation |
| DiT-XL/2 | `DiT-XL-2-256x256.pt` + `sd-vae-ft-ema` | DDIM-50, guidance 1.5 or off |
| StyleGAN2-ADA | CelebA-HQ 256 pickle (G_ema) | one pass |
| DDPM | `google/ddpm-celebahq-256` | DDIM, 100 steps |
| R2 backbone | MAE ViT-B/16 (frozen) | mean of the 196 final-layer patch tokens |

The external code bases (improved-diffusion, EDM, EDM2, StyleGAN-XL, DiT, StyleGAN2-ADA and
`diffusers`) are expected under `$PROJECT_ROOT/external`.

## Environment

Python 3.10+ with PyTorch, NumPy, SciPy, pandas and matplotlib. Paths are read from
environment variables:

| Variable | Meaning |
|---|---|
| `PROJECT_ROOT` | working directory for sample caches, intermediate arrays and outputs |
| `DATA_ROOT` | ImageNet-32 arrays |
| `CACHE_ROOT` | large caches and optional pinned dependencies |
| `HF_CACHE` | Hugging Face cache for checkpoints |
| `WRAPFID_ROOT`, `WRAPFID_STORE` | working and storage directories of the wrapping/FID study |
| `NUM_THREADS` | CPU threads for the severity computations (default: all) |

Custom CUDA extensions of the StyleGAN code bases are optional; without them the upstream
reference operators give the same sample distribution but not bit-identical samples.

## Order of the full run

1. `functionals/build_calibration_law.py` – PCA chart and calibration law on the calibration split
2. `audits/audit_*.py` – sample each model and audit it under R1
3. `functionals/r2_*.py`, `functionals/r3_audit.py`, `functionals/concordance.py` – R2, R3 and their agreement
4. `validation/*.py` – self-validation, planted ratios, sensitivity, ablations
5. `experiments/*.py` – mixtures, quality axis, shape test, wrapper, downstream estimates, tables
6. `celebahq/*.py` – replication
7. `wrap_fid/sample_*.py`, `wrap_and_measure.py`, `bootstrap_fid.py`, `tail_quality.py`, `guard.py` – FID study
8. figure scripts as above
