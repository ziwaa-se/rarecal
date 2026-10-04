"""Produce the result figures from the result CSVs."""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TAILAUDIT_DATA", os.path.join(HERE, "..", "..", "results"))
FIGS = os.environ.get("TAILAUDIT_FIGS", os.path.join(HERE, "..", "figures"))
os.makedirs(FIGS, exist_ok=True)

INK, MUTED, SURFACE = "#33322e", "#8a887f", "#ffffff"
CONFIGS = ["iDDPM", "EDM", "EDM-churn", "EDM2-S", "EDM2-M", "StyleGAN-XL",
           "DiT-cfg1.5", "DiT-cfg1.0"]
PAL8 = {"iDDPM": "#8a4f9e", "EDM": "#2a78d6", "EDM-churn": "#eb6834",
        "EDM2-S": "#2a9d64", "EDM2-M": "#8f2358", "StyleGAN-XL": "#b8860b",
        "DiT-cfg1.5": "#3a6db8", "DiT-cfg1.0": "#d06aa8"}
MRK8 = {"iDDPM": "o", "EDM": "s", "EDM-churn": "^", "EDM2-S": "D",
        "EDM2-M": "v", "StyleGAN-XL": "P", "DiT-cfg1.5": "X", "DiT-cfg1.0": "*"}
FCOL = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#2a9d64"}
FMRK = {"S1": "o", "S2": "s", "S3": "^"}

plt.rcParams.update({
    "font.size": 8.5, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def rd(name):
    return list(csv.DictReader(open(f"{DATA}/{name}")))


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


F = float

# ============================ fig_scatter.pdf ================================
lb = [r for r in rd("leaderboard_r1.csv") if F(r["p"]) == 1e-3]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for r in lb:
    if not r["fid_published"]:
        continue
    fid, rho = F(r["fid_published"]), F(r["rho"])
    lo, hi = F(r["quad_lo"]), F(r["quad_hi"])
    c = PAL8[r["config"]]
    ax.errorbar(fid, rho, yerr=[[rho - lo], [hi - rho]], fmt=MRK8[r["config"]],
                ms=6, color=c, capsize=2.5, lw=1.2, mew=0.8)
    dx, dy = (6, 5)
    if r["config"] == "EDM2-M":
        dx, dy = (6, -11)
    if r["config"] == "DiT-cfg1.5":
        dx, dy = (-4, 8)
    ax.annotate(r["config"], (fid, rho), textcoords="offset points",
                xytext=(dx, dy), fontsize=7.5, color=INK)
ax.axhline(1.0, color=INK, lw=0.9, ls="--")
ax.text(0.98, 1.0, "calibrated", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7.5, color=MUTED)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("FID (ImageNet, published)  →  better")
ax.set_ylabel(r"tail-calibration ratio  $\hat\rho_{R_1}(10^{-3})$")
ax.set_title("Tail-calibration ratio against published FID\n"
             "(severity functional $R_1$; 95% quadrature intervals)",
             loc="left", fontweight="bold")
ax.text(0.02, 0.03, "EDM-churn omitted (no published FID); DiT rows: final "
        "M=50k caches", transform=ax.transAxes, fontsize=6.5, color=MUTED)
ax.grid(alpha=0.25, lw=0.5)
save(fig, "fig_scatter.pdf")

# ============================ fig_depth.pdf ==================================
dep = rd("depth_scan.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for cfg in CONFIGS:
    rr = [r for r in dep if r["config"] == cfg]
    ps = np.array([F(r["p"]) for r in rr])
    rho = np.array([F(r["rho"]) for r in rr])
    lo = np.array([F(r["quad_lo"]) for r in rr])
    hi = np.array([F(r["quad_hi"]) for r in rr])
    ax.plot(ps, np.where(rho > 0, rho, np.nan), color=PAL8[cfg], lw=1.4,
            marker=MRK8[cfg], ms=3.5, label=cfg, mew=0.6)
    ax.fill_between(ps, np.maximum(lo, 1e-2), hi, color=PAL8[cfg], alpha=0.10,
                    lw=0)
ax.axhline(1.0, color=INK, lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("event level p  →  rarer")
ax.set_ylabel(r"$\hat\rho_{R_1}(p)$, 95% quadrature band")
ax.set_title("Tail-calibration ratio by depth ($R_1$)",
             loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.9,
          handletextpad=0.4)
ax.grid(alpha=0.25, lw=0.5)
save(fig, "fig_depth.pdf")


# ============================ fig_recalibration.pdf ===========================
wi = rd("wrapped_independence.csv")
lb_all = rd("leaderboard_r1.csv")
raw = {(r["config"], F(r["p"])): (F(r["rho"]), F(r["wilson_lo"]),
                                  F(r["wilson_hi"])) for r in lb_all}
wrap_by_cfg = {r["model"]: (F(r["rho3_wrapped"]), F(r["rho4_wrapped"]))
               for r in wi}
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4), sharex=True,
                        gridspec_kw=dict(wspace=0.42))
xpos = np.arange(len(CONFIGS))
for ax, p, wl, elab in [(axs[0], 1e-3, 0.990, r"$=e_{\rm grid}(10^{-3})$"),
                        (axs[1], 1e-4, 0.600, r"$=e_{\rm grid}(10^{-4})$")]:
    for i, cfg in enumerate(CONFIGS):
        rho, lo, hi = raw[(cfg, p)]
        ax.errorbar(i - 0.12, rho, yerr=[[rho - lo], [hi - rho]],
                    fmt=MRK8[cfg], ms=5, color=PAL8[cfg], capsize=2, lw=1.1,
                    zorder=3)
        wv = wrap_by_cfg[cfg][0 if p == 1e-3 else 1]
        ax.plot(i + 0.18, wv, marker="_", ms=9, mew=2.2, color=INK, zorder=4)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.axhline(wl, color=INK, lw=1.0, ls="-", alpha=0.75)
    ax.text(1.015, wl, f"wrapped\n{wl:.3f} {elab}",
            transform=ax.get_yaxis_transform(), fontsize=6.6, va="center",
            ha="left", color=INK)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(CONFIGS, rotation=38, ha="right", fontsize=7)
    ax.set_title(f"depth p = {p:g}", loc="left")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
axs[0].set_ylabel(r"$\hat\rho(p)$")
h1 = plt.Line2D([], [], marker="o", ls="", color=MUTED, label="raw (95% Wilson interval)")
h2 = plt.Line2D([], [], marker="_", ls="", mew=2.2, color=INK,
                label="wrapped (line: M=100k; dashes: value at each M)")
fig.legend(handles=[h1, h2], frameon=False, loc="lower center", ncol=2,
           fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Raw and recalibrated tail-calibration ratios",
             fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.90])
save(fig, "fig_recalibration.pdf")

# ============================ fig_concordance.pdf ============================
fun = rd("leaderboard_functional.csv")
fig, ax = plt.subplots(figsize=(5.0, 4.4))
ylab = []
for i, cfg in enumerate(CONFIGS):
    y0 = len(CONFIGS) - 1 - i
    ylab.append((y0, cfg))
    for j, f_ in enumerate(["S1", "S2", "S3"]):
        r = next(x for x in fun if x["config"] == cfg and x["functional"] == f_)
        rho, lo, hi = F(r["rho"]), F(r["quad_lo"]), F(r["quad_hi"])
        y = y0 + (1 - j) * 0.22
        ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]], fmt=FMRK[f_],
                    ms=4.5, color=FCOL[f_], capsize=2, lw=1.1,
                    label={"S1": "$R_1$", "S2": "$R_2$", "S3": "$R_3$"}.get(f_, f_) if i == 0 else None)
        if cfg == "EDM-churn" and f_ == "S2":
            ax.annotate("M=15k", (hi, y), textcoords="offset points",
                        xytext=(3, -2), fontsize=6, color=MUTED)
ax.axvline(1.0, color=INK, lw=0.9, ls="--")
ax.set_yticks([y for y, _ in ylab])
ax.set_yticklabels([c for _, c in ylab], fontsize=8)
ax.set_xscale("log")
ax.set_xlabel(r"$\hat\rho(10^{-3})$, 95% quadrature certificates")
ax.set_title("Tail-calibration ratio under three severity functionals",
             loc="left", fontweight="bold")
iy = len(CONFIGS) - 1
ax.annotate("iDDPM:\n$R_1$ under (0.17), $R_2$ over (1.55)",
            xy=(0.45, iy - 0.05), fontsize=7, color=INK,
            ha="center", va="top",
            bbox=dict(fc="#f6f2ea", ec=MUTED, lw=0.5, pad=2.5))
ax.legend(frameon=False, loc="lower right", title=None)
ax.grid(alpha=0.25, lw=0.5, axis="x")
save(fig, "fig_concordance.pdf")

# ============================ fig_planted_truth.pdf ==========================
pt = rd("planted_ratio.csv")
ptq = rd("planted_ratio_coverage.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axs[0]
for var, mk, col in [("A_fixed_tau", "o", "#2a78d6"),
                     ("B_split_tau", "s", "#eb6834")]:
    for p, alpha in [(1e-2, 1.0), (1e-3, 0.55)]:
        rr = [r for r in pt if r["variant"] == var and F(r["p"]) == p]
        x = [F(r["rho_true"]) for r in rr]
        y = [F(r["mean_rho_hat"]) for r in rr]
        ax.plot(x, y, mk, ms=5, color=col, alpha=alpha, ls="",
                label=f"{'A (corpus-relative)' if var[0]=='A' else 'B (independent threshold)'}"
                      f", p={p:g}")
g = np.array([0.25, 6])
ax.plot(g, g, ls="--", lw=0.9, color=INK)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"planted $\rho$")
ax.set_ylabel(r"mean audited $\hat\rho$ (B=500)")
ax.set_title("Recovery of the planted ratio", loc="left")
ax.legend(frameon=False, fontsize=6.8, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)
ax = axs[1]
cells = [(F(r["p"]), F(r["rho_target"])) for r in ptq]
xpos = np.arange(len(ptq))
wid = 0.38
ax.bar(xpos - wid / 2, [F(r["coverage_wilson"]) for r in ptq], wid,
       color="#eb6834", label="Wilson only")
ax.bar(xpos + wid / 2, [F(r["coverage_quadrature"]) for r in ptq], wid,
       color="#2a78d6", label="quadrature, Eq. (1)")
ax.axhline(0.95, color=INK, lw=0.9, ls="--")
ax.text(0.02, 0.952, "nominal 0.95", transform=ax.get_yaxis_transform(),
        fontsize=6.5, va="bottom", color=INK)
ax.set_xticks(xpos)
ax.set_xticklabels([f"p={p:g}\nrho={r:g}" for p, r in cells], fontsize=6.3)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("interval coverage of the planted ratio")
ax.set_title("Coverage, independent-threshold design", loc="left")
ax.legend(frameon=False, fontsize=6.8, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), ncol=2)
ax.grid(alpha=0.25, lw=0.5, axis="y")
fig.suptitle("Planted-ratio recovery and interval coverage "
             "(B=500 per cell, M=100k)", fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.92])
save(fig, "fig_planted_truth.pdf")

# ============================ fig_downstream.pdf =========================
e2 = rd("downstream_inheritance.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=False)
for ax, p in zip(axs, [1e-3, 1e-4]):
    for model in ["iDDPM", "EDM"]:
        col = PAL8[model]
        for arm, ls in [("raw", (0, (5, 2))), ("wrapped", "-")]:
            rr = [r for r in e2 if r["model"] == model and r["arm"] == arm
                  and F(r["p"]) == p]
            ns = [F(r["n"]) for r in rr]
            mu = np.array([F(r["mean_ratio"]) for r in rr])
            sd = np.array([F(r["sd_ratio"]) for r in rr])
            ax.plot(ns, mu, linestyle=ls, color=col, lw=1.5,
                    marker=MRK8[model], ms=3.2,
                    label=f"{model} {arm}")
            ax.fill_between(ns, mu - sd, mu + sd, color=col, alpha=0.10, lw=0)
    rr = [r for r in e2 if r["arm"] == "real-control" and F(r["p"]) == p]
    ax.plot([F(r["n"]) for r in rr], [F(r["mean_ratio"]) for r in rr], ":",
            color=INK, lw=1.3, label="real-data control")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("synthetic corpus size n")
    ax.set_title(f"depth p = {p:g}", loc="left")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel(r"$\hat p_n / p$")
hh, ll = axs[0].get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, fontsize=7, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.035), handlelength=4.0)
fig.suptitle("Downstream plug-in estimates by corpus size",
             fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
save(fig, "fig_downstream.pdf")

# ============================ fig_scale_curves.pdf ===========================
scv = rd("scale_curves.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
for ax, p in zip(axs, [1e-3, 1e-4]):
    for cfg in CONFIGS:
        rr = [r for r in scv if r["config"] == cfg and F(r["p"]) == p]
        if not rr:
            continue
        ns = [F(r["M_sub"]) for r in rr]
        w = [F(r["mean_ci_width"]) for r in rr]
        ax.plot(ns, w, "-", color=PAL8[cfg], lw=1.3, marker=MRK8[cfg], ms=3.2,
                label=cfg)
    ax.axvline(50_000, color=MUTED, lw=0.9, ls=":")
    ax.text(50_000, 0.97, "community-standard 50k",
            transform=ax.get_xaxis_transform(), rotation=90, fontsize=6.3,
            color=MUTED, ha="right", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("audit sample budget M'")
    ax.set_title(f"depth p = {p:g}", loc="left")
    ax.grid(alpha=0.25, lw=0.5)
w50 = [F(r["mean_ci_width"]) for r in scv
       if F(r["p"]) == 1e-4 and r["M_sub"] == "50000"]
axs[1].annotate(f"M'=50k: mean 95% width\n"
                f"{min(w50):.2f}–{max(w50):.2f} $\\rho$-units",
                xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
                color=INK, bbox=dict(fc="#f6f2ea", ec=MUTED, lw=0.5, pad=3))
axs[0].set_ylabel(r"mean Wilson CI width ($\rho$ units)")
hh, ll = axs[0].get_legend_handles_labels()
fig.legend(hh, ll, frameon=False, fontsize=6.8, loc="lower center", ncol=8,
           columnspacing=0.7, handletextpad=0.35, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("Interval width by sampling budget",
             fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
save(fig, "fig_scale_curves.pdf")

# ============================ fig_shape_collapse.pdf =========================
sh = [x for x in rd("shape_dispersion.csv") if x["seed"] == "20260816"]
# Holm correction of the Brown-Forsythe p-values over all 32 cells
_ord = sorted(range(len(sh)), key=lambda i: F(sh[i]["p_bf"]))
_bf_rej = set()
for _k, _i in enumerate(_ord):
    if F(sh[_i]["p_bf"]) <= 0.05 / (len(sh) - _k):
        _bf_rej.add(_i)
    else:
        break
for _i, _x in enumerate(sh):
    _x["bf_reject"] = "1" if _i in _bf_rej else "0"
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, q in zip(axs, [1e-3, 1e-2]):
    for i, cfg in enumerate(CONFIGS):
        y0 = len(CONFIGS) - 1 - i
        for pc, dy, alpha in [("PC1", 0.14, 1.0), ("PC2", -0.14, 0.45)]:
            r = next(x for x in sh if x["config"] == cfg
                     and F(x["depth"]) == q and x["pc"] == pc)
            ratio, lo, hi = F(r["sd_ratio"]), F(r["ratio_lo"]), F(r["ratio_hi"])
            rej = r["bf_reject"] == "1"
            ax.errorbar(ratio, y0 + dy, xerr=[[ratio - lo], [hi - ratio]],
                        fmt="o", ms=4.5 if pc == "PC1" else 3.2,
                        color=PAL8[cfg], alpha=alpha, capsize=2, lw=1.0,
                        mfc=PAL8[cfg] if rej else "white", mew=1.0)
    ax.axvline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xlim(0.25, 1.35)
    ax.set_xlabel(r"tail dispersion ratio  $\hat\sigma_{gen}/\hat\sigma_{real}$")
    ax.set_title(f"depth q = {q:g}", loc="left")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
axs[0].set_yticks(range(len(CONFIGS)))
axs[0].set_yticklabels(CONFIGS[::-1], fontsize=8)
h1 = plt.Line2D([], [], marker="o", ls="", color=INK, label="Brown-Forsythe, Holm-rejected (p<.05)")
h2 = plt.Line2D([], [], marker="o", ls="", color=INK, mfc="white",
                label="not rejected")
h3 = plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=3.2, alpha=0.45,
                label="PC2 (smaller)")
fig.legend(handles=[h1, h2, h3], frameon=False, fontsize=7,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Conditional tail dispersion relative to real data "
             "(calibration-tail PC basis; bootstrap 95% intervals)", fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.07, 1, 0.93])
save(fig, "fig_shape_collapse.pdf")

# ============================ fig_sensitivity.pdf ===========================
e5 = rd("sensitivity.csv")
VARS = ["default", "K30", "K100", "seed8001", "seed8002", "fixedq0.98",
        "fixedq0.995"]
VCOL = {"default": INK, "K30": "#8a4f9e", "K100": "#2a78d6",
        "seed8001": "#2a9d64", "seed8002": "#7cb87f", "fixedq0.98": "#eb6834",
        "fixedq0.995": "#8f2358"}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, cfg in enumerate(CONFIGS):
    for j, v in enumerate(VARS):
        r = next((x for x in e5 if x["config"] == cfg and x["variant"] == v),
                 None)
        if r is None:
            continue
        x = i + (j - 3) * 0.105
        rho, lo, hi = F(r["rho"]), F(r["lo"]), F(r["hi"])
        big = v == "default"
        ax.errorbar(x, rho, yerr=[[rho - lo], [hi - rho]], fmt="o",
                    ms=5.5 if big else 3, color=VCOL[v], capsize=0,
                    lw=1.6 if big else 1.0, label=v if i == 0 else None)
ax.axhline(1.0, color=INK, lw=0.9, ls="--")
ax.set_yscale("log")
ax.set_xticks(range(len(CONFIGS)))
ax.set_xticklabels(CONFIGS, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel(r"$\hat\rho(10^{-3})$, 95% Wilson CI")
ax.set_title("Sensitivity to basis dimension, split seed and threshold rule",
             loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=4, fontsize=6.8, loc="upper right")
ax.grid(alpha=0.25, lw=0.5, axis="y")
save(fig, "fig_sensitivity.pdf")

print("ALL FIGURES DONE", flush=True)
