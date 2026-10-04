"""Produce the tables and the summary figure of the wrapping/FID study from the result CSVs."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results", "wrap_fid")
main = pd.read_csv(f"{R}/wrapfid_main.csv")
abl = pd.read_csv(f"{R}/wrapfid_ablation.csv")
boot = pd.read_csv(f"{R}/bootstrap_fid.csv")
tq = pd.read_csv(f"{R}/tailquality.csv")
guard = pd.read_csv(f"{R}/guard_results.csv")
allarms = pd.concat([main, abl], ignore_index=True)
CONFIG_LIST = ["iDDPM", "EDM", "EDM2-S", "EDM2-M", "StyleGAN-XL", "DiT-cfg1.5", "DiT-cfg1.0"]
ORDER = CONFIG_LIST + ["REAL-POOL"]
LAB = {"REAL-POOL": r"Real images$^{\star}$"}
ARROW = {"under": r"^{\downarrow}", "over": r"^{\uparrow}", "calibrated": ""}


def g(df, config, arm=None, col=None, armcol="arm"):
    s = df[df.config == config]
    if arm is not None:
        s = s[s[armcol] == arm]
    assert len(s) == 1, (config, arm, len(s))
    return s.iloc[0] if col is None else s.iloc[0][col]


def rho(r, p="3"):
    return f"${r[f'rho{p}_img']:.2f}{ARROW[r[f'rho{p}_verdict']]}$ [{r[f'rho{p}_lo']:.2f}, {r[f'rho{p}_hi']:.2f}]"


def write(name, body):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(body)
    print("wrote", name)


# ---------------------------------------------------------------- Table 1: main
rows = []
for c in ORDER:
    a, b = g(main, c, "raw"), g(main, c, "wrap-tail")
    bt = g(boot, c, "wrap-tail")
    rows.append(f"{LAB.get(c, c)} & {int(a.M/1000)}k & {a.fid_sub:.3f} & {b.fid_sub:.3f} & "
                f"${bt.dfid_boot_mean:+.4f}$ [${bt.dfid_ci_lo:+.4f}$, ${bt.dfid_ci_hi:+.4f}$] & {bt.fid_raw_boot_sd:.3f} & "
                f"{rho(a)} & {rho(b)} & ${b.rho4_img:.2f}$ \\\\")
write("tab_wrapfid_main.tex", r"""% Source: results/wrapfid_main.csv (fid_sub, rho3_*, rho4_img; arm raw vs wrap-tail),
% results/bootstrap_fid.csv (arm wrap-tail: dfid_boot_mean, dfid_ci_lo/hi, fid_raw_boot_sd).
% FID = Inception-v3 pool3, torch-fidelity recipe, 32->299 upsampling, n=50k fixed paired subsample,
% reference = 540,583 held-out ImageNet-32 images (REF half of the hold split).
\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\caption{The wrapper repairs the tail rate at no measurable cost in FID. FID is measured here (Inception-v3,
$n=50$k, one fixed real reference of $540{,}583$ held-out images); $\Delta$FID is the paired-bootstrap mean
($B=200$) with its $95\%$ interval, and ``sd'' is the resampling standard deviation of the raw FID itself.
$\hat\rho$ is read off the \emph{realized uint8 images} with the quadrature certificate of
Eq.~(\ref{eq:quad}). $^{\star}$\,$100$k held-out real images in the sampler role.}
\label{tab:wrapfid}
\begin{tabular}{@{}lrrrlrllr@{}}
\toprule
& & \multicolumn{4}{c}{FID (Inception, $32^2$)} & \multicolumn{3}{c}{$\hat\rho_{\sevone}$ at depth}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}
Configuration & $M$ & raw & wrapped & $\Delta$FID [95\%] & sd & raw, $10^{-3}$ & wrapped, $10^{-3}$ & wr., $10^{-4}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table 2: ablation
rows = []
for c in ORDER:
    t, f, h = g(main, c, "wrap-tail"), g(abl, c, "wrap-full"), g(abl, c, "wrap-tail-hardclip")
    bf, bh = g(boot, c, "wrap-full"), g(boot, c, "wrap-tail-hardclip")
    rows.append(f"{LAB.get(c, c)} & ${bf.dfid_boot_mean:+.3f}$ [${bf.dfid_ci_lo:+.3f}$, ${bf.dfid_ci_hi:+.3f}$] & "
                f"{100*f.frac_modified:.0f}\\% & {rho(f)} & ${bh.dfid_boot_mean:+.4f}$ & {rho(h)} & "
                f"{100*h.sev_u8_within_005:.0f}\\% & {100*t.sev_u8_within_005:.0f}\\% \\\\")
write("tab_wrapfid_ablation.tex", r"""% Source: results/wrapfid_ablation.csv (arms wrap-full, wrap-tail-hardclip), results/wrapfid_main.csv
% (arm wrap-tail, last column), results/bootstrap_fid.csv (dfid_boot_*).
\begin{table}[h]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Two ablations of the realization. \emph{Full wrap} applies Algorithm~\ref{alg:wrap} to all $M$ samples rather than
the top $1\%$ by rank: identical tail rates, a measurable FID cost. \emph{Hard clip} replaces the damped alternating
projection by one radial step and one clip: identical FID, but truncated samplers are not repaired. ``within''
is the share of transported samples whose realized severity lands within $0.05$ of its target.}
\label{tab:wrapfid-ablation}
\begin{tabular}{@{}llrllllr@{}}
\toprule
& \multicolumn{3}{c}{full wrap} & \multicolumn{3}{c}{hard clip} & damped\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}
Configuration & $\Delta$FID [95\%] & modified & $\hat\rho(10^{-3})$ & $\Delta$FID & $\hat\rho(10^{-3})$ & within & within\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table 3: tail image quality
rows = []
for c in ORDER:
    r = g(tq, c)
    rows.append(f"{LAB.get(c, c)} & {r.n_changed}/{r.n_touched} & {r.mean_abs_pix_med:.2f} & {r.psnr_med:.1f} & "
                f"{r.ssim_med:.3f} / {r.ssim_p05:.3f} & {r.lpips_med:.4f} / {r.lpips_p95:.4f} & "
                f"{100*r.sat_before_mean:.1f} $\\to$ {100*r.sat_after_mean:.1f} & "
                f"{r.precision_before:.3f} $\\to$ {r.precision_after:.3f} & {r.precision_bulk_same_model:.3f} & "
                f"{r.tailkid_before:.4f} $\\to$ {r.tailkid_after:.4f} \\\\")
write("tab_wrapfid_tailquality.tex", r"""% Source: results/tailquality.csv. Touched set = the top 1% of each sampler by severity rank (the only images
% the wrapper changes). Precision = improved precision (Kynkaanniemi et al. 2019), k=3, Inception pool3 space,
% reference manifold = 50,000 random REF reals + the 5,406 REF real-tail images (R >= 11.76).
\begin{table}[h]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{The images the wrapper touches remain usable. Per-image change of the transported samples (medians, with the
$5$th/$95$th percentile where shown), share of pixels on the box boundary before and after, improved precision of the
\emph{same} samples before and after against a real-image manifold (last precision column: the same sampler's untouched
samples), and KID of the touched set to the real tail. The held-out real tail itself scores $0.658$.}
\label{tab:wrapfid-tailq}
\begin{tabular}{@{}lrrrllllrl@{}}
\toprule
Configuration & changed & $|\Delta\mathrm{pix}|$ & PSNR & SSIM med / p05 & LPIPS med / p95 & sat.\ \% & precision & bulk & tail KID\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table 4: guard
rows = []
for c in ["iDDPM", "EDM", "DiT-cfg1.0", "REAL-POOL"]:
    for v, vl in [("wrap-tail", "none"), ("guard-res", "residual"), ("guard-res+ssim", "residual or SSIM")]:
        r = g(guard, c, v, armcol="variant")
        rows.append(f"{LAB.get(c, c)} & {vl} & {r.n_reverted} & {r.ssim_min:.2f} & {r.n_ssim_lt_080} & "
                    f"${r.dfid_boot_mean:+.4f}$ & {rho(r)} & ${r.rho4_img:.2f}$ & {r.precision_after:.3f} \\\\")
    rows.append(r"\addlinespace[1pt]")
write("tab_wrapfid_guard.tex", r"""% Source: results/guard/guard_results.csv. The four configurations not listed (EDM2-S, EDM2-M, StyleGAN-XL,
% DiT-cfg1.5) revert zero samples under both guards and are numerically identical to Table tab:wrapfid.
\begin{table}[h]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{An optional per-sample guard: a transported image is replaced by the original sample when the projection
did not converge (severity residual $>0.05$) or, in the second variant, when it also altered the image visibly
(SSIM $<0.80$). $21$ images are reverted in total across all eight sample sets; no verdict changes.}
\label{tab:wrapfid-guard}
\begin{tabular}{@{}llrrrrlrr@{}}
\toprule
Configuration & guard & reverted & min SSIM & \#SSIM$<0.8$ & $\Delta$FID & $\hat\rho(10^{-3})$ & $\hat\rho(10^{-4})$ & precision\\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table 5: measured FID of the audited configurations
rows = []
pub = {"iDDPM": "2.90", "EDM": "2.60", "EDM2-S": "1.58", "EDM2-M": "1.43", "StyleGAN-XL": "1.10",
       "DiT-cfg1.5": "2.27", "DiT-cfg1.0": "9.60"}
for c in CONFIG_LIST:
    a = g(main, c, "raw")
    rows.append(f"{c} & {pub[c]} & {a.fid_full:.2f} & {a.fid_sub:.2f} & {a.kid_mean*1e3:.2f} & {a.fmd_full:.3f} & {rho(a)} \\\\")
a = g(main, "REAL-POOL", "raw")
rows.append(r"\addlinespace[1pt]" + "\n" + f"{LAB['REAL-POOL']} & -- & {a.fid_full:.2f} & {a.fid_sub:.2f} & {a.kid_mean*1e3:.2f} & {a.fmd_full:.3f} & {rho(a)} \\\\")
write("tab_zoo_fid32.tex", r"""% Source: results/wrapfid_main.csv, arm raw (fid_full at the audited M, fid_sub at n=50k, kid_mean, fmd_full).
% Published FID column as reported by each model's authors (own resolution and protocol).
\begin{table}[h]
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{FID measured on the audited $32^2$ sample sets (Inception-v3 pool3, torch-fidelity recipe, one
fixed reference of $540{,}583$ held-out images), next to the published values and the MAE-space FMD of
Table~\ref{tab:leaderboard}. KID is $\times10^{3}$.}
\label{tab:zoofid}
\begin{tabular}{@{}lrrrrrl@{}}
\toprule
Configuration & FID publ. & FID ($M$) & FID ($50$k) & KID & FMD & $\hat\rho_{\sevone}(10^{-3})$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- summary figure (two panels, one y-axis each)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), facecolor=SURF)
x = np.arange(len(CONFIG_LIST))
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LIST, rotation=28, ha="right")
for off, arm, col, lab in [(-0.14, "raw", ORANGE, "raw sampler"), (0.14, "wrap-tail", BLUE, "wrapped")]:
    r = [g(main, c, arm) for c in CONFIG_LIST]
    y = np.array([q.rho3_img for q in r])
    lo = y - np.array([q.rho3_lo for q in r])
    hi = np.array([q.rho3_hi for q in r]) - y
    ax1.errorbar(x + off, y, yerr=[lo, hi], fmt="o", ms=4.2, lw=1.1, capsize=0, color=col, label=lab,
                 markeredgecolor=SURF, markeredgewidth=0.8)
ax1.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax1.set_yscale("log")
ax1.set_yticks([0.1, 0.2, 0.5, 1, 2])
ax1.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
ax1.set_ylabel(r"$\hat\rho_{R_1}(10^{-3})$, realized images")
ax1.set_title("Tail rate: repaired", loc="left", fontsize=8.5, fontweight="bold")
ax1.legend(frameon=False, loc="lower right", fontsize=7.5)
sd = np.array([g(boot, c, "wrap-tail").fid_raw_boot_sd for c in CONFIG_LIST])
ax2.bar(x, 2 * sd, bottom=-sd, width=0.72, color="#ecebe7", zorder=0, label=r"$\pm1$ sd of raw FID (resampling)")
for off, arm, col, lab in [(-0.14, "wrap-tail", BLUE, "wrapper (top 1%)"), (0.14, "wrap-full", ORANGE, "full wrap (all $M$)")]:
    r = [g(boot, c, arm) for c in CONFIG_LIST]
    y = np.array([q.dfid_boot_mean for q in r])
    ax2.errorbar(x + off, y, yerr=[y - np.array([q.dfid_ci_lo for q in r]), np.array([q.dfid_ci_hi for q in r]) - y],
                 fmt="o", ms=4.2, lw=1.1, capsize=0, color=col, label=lab, markeredgecolor=SURF, markeredgewidth=0.8)
ax2.axhline(0.0, color=MUTED, lw=0.9)
ax2.set_ylabel(r"$\Delta$FID (wrapped $-$ raw), $n=50$k")
ax2.set_title("FID: unchanged by the wrapper", loc="left", fontsize=8.5, fontweight="bold")
ax2.set_ylim(-0.05, 0.33)
ax2.legend(frameon=False, loc="upper left", fontsize=7.2)
fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(HERE, "fig_wrapfid_summary.pdf"), facecolor=SURF)
fig.savefig(os.path.join(HERE, "fig_wrapfid_summary.png"), dpi=220, facecolor=SURF)
print("wrote fig_wrapfid_summary.{pdf,png}")

# ---------------------------------------------------------------- flat number dictionary
N = {}
for _, r in allarms.iterrows():
    k = f"{r.config}|{r.arm}"
    for col in ["M", "fid_sub", "fid_full", "kid_mean", "fmd_sub", "fmd_full", "rho3_raw", "rho3_remap", "rho3_img", "rho3_lo",
                "rho3_hi", "rho3_verdict", "k3", "rho4_img", "rho4_lo", "rho4_hi", "rho4_verdict", "k4", "n_transported",
                "n_modified", "frac_modified", "mean_abs_pix_modified", "oor0", "sev_within_005", "sev_u8_within_005", "sev_err_max"]:
        v = r[col]
        N[f"{k}|{col}"] = v if isinstance(v, str) else (None if pd.isna(v) else float(v))
for _, r in boot.iterrows():
    for col in ["dfid_boot_mean", "dfid_boot_sd", "dfid_ci_lo", "dfid_ci_hi", "fid_raw_boot_sd", "frac_positive", "ratio_absdelta_to_rawsd"]:
        N[f"{r.config}|{r.arm}|boot|{col}"] = float(r[col])
for _, r in tq.iterrows():
    for col in tq.columns[1:]:
        N[f"{r.config}|tailquality|{col}"] = float(r[col])
for _, r in guard.iterrows():
    for col in ["n_reverted", "ssim_min", "n_ssim_lt_080", "dfid_boot_mean", "rho3_img", "rho3_lo", "rho3_hi", "rho4_img", "precision_after"]:
        N[f"{r.config}|{r.variant}|guard|{col}"] = float(r[col])
t = boot[boot.arm == "wrap-tail"]
z = t[t.config.isin(CONFIG_LIST)]
N["SUMMARY|max_abs_dfid_tail_configs"] = float(z.dfid_boot_mean.abs().max())
N["SUMMARY|max_ratio_dfid_to_rawsd_tail_configs"] = float(z.ratio_absdelta_to_rawsd.max())
f = boot[(boot.arm == "wrap-full") & boot.config.isin(CONFIG_LIST)]
N["SUMMARY|min_dfid_full_configs"], N["SUMMARY|max_dfid_full_configs"] = float(f.dfid_boot_mean.min()), float(f.dfid_boot_mean.max())
N["SUMMARY|total_reverted_guard_res+ssim"] = float(guard[guard.variant == "guard-res+ssim"].n_reverted.sum())
with open(os.path.join(HERE, "all_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote all_numbers.json with", len(N), "entries")
for k in [k for k in N if k.startswith("SUMMARY")]:
    print("  ", k, "=", N[k])
