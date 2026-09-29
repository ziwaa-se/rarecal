"""Figure for Proposition 1: schematic, measured construction, and comparison scales."""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA, FIGS = f"{BASE}/data", f"{BASE}/figures"
INK, MUTED, SURFACE = "#33322e", "#8a887f", "#ffffff"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2a9d64"
plt.rcParams.update({
    "font.size": 8.5, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "pdf.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE})

rows = list(csv.DictReader(open(f"{DATA}/prop1_empirical.csv")))
qa = list(csv.DictReader(open(f"{DATA}/quality_axis_fmd.csv")))
con = {r["bulk_law"]: r for r in csv.DictReader(
    open(f"{DATA}/prop1_construction.csv"))}

F = float
BASE_CFG = "EDM"
# Panel B plots the arm that matches EXACTLY the moment map the Caratheodory
# remark is about -- the mean map psi = phi -- i.e. the Frank-Wolfe witness.
# Panel C then shows the whole family, including the second-moment-matched arm
# and the UNMATCHED control, so nothing is selected on the outcome.
LAW = "sparse"
LAW2 = "mom2"
for k in con:
    print(f"  Q_B[{k:7s}]: |dmu|/|mu_T| = "
          f"{F(con[k]['resid_mean_rel_to_muT']):.3e}, |dS|_F/|S_T|_F = "
          f"{F(con[k]['resid_2ndmoment_full_rel']):.3e}, support "
          f"{con[k]['n_support']}")

fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(13.0, 3.7),
    gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.25]))

# ---------------- Panel A: the Caratheodory picture (illustrative) ---------
rng = np.random.default_rng(0)
bulk = rng.normal(0, 1.0, size=(220, 2))
bulk = bulk[np.linalg.norm(bulk, axis=1) < 2.0]
ang = rng.uniform(0, 2 * np.pi, 14)
rad = rng.uniform(2.6, 3.2, 14)
tail = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
axA.scatter(*bulk.T, s=9, color=MUTED, alpha=0.55, linewidths=0,
            label=r"bulk features $\psi(B_\tau)$")
axA.scatter(*tail.T, s=26, color=ORANGE, alpha=0.95, linewidths=0,
            label=r"tail features $\psi(T_\tau)$")
target = tail.mean(axis=0)
d = np.linalg.norm(bulk - target, axis=1)
hull = bulk[np.argsort(d)[[0, 3, 9]]]
w = np.linalg.solve(np.vstack([hull.T, np.ones(3)]), np.r_[target, 1.0])
assert np.allclose(w @ hull, target, atol=1e-9)
axA.add_patch(plt.Polygon(hull, closed=True, fill=True, facecolor=BLUE,
                          alpha=0.13, edgecolor=BLUE, lw=1.2, zorder=1))
axA.scatter(*hull.T, s=34, facecolor="none", edgecolor=BLUE, lw=1.3, zorder=3)
axA.scatter(*target, s=95, marker="*", color=INK, zorder=4,
            label=r"$\mathbb{E}_{Q_T}\psi=\mathbb{E}_{Q_B}\psi$")
axA.annotate(r"hull of $k{+}1$ bulk points", xy=hull.mean(axis=0),
             xytext=(-3.0, 3.05), fontsize=7.6, color=BLUE,
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
axA.set_xticks([]); axA.set_yticks([])
axA.set_xlim(-3.6, 3.6); axA.set_ylim(-3.6, 3.6)
axA.set_title("Moment mixability (illustrative)", fontweight="bold",
              loc="left")
axA.set_xlabel(r"feature coordinate $\psi_1$")
axA.set_ylabel(r"feature coordinate $\psi_2$")
axA.legend(loc="lower left", frameon=False)

# ---------------- Panel B: the construction, measured ----------------------
configs = sorted([(F(r["fmd_mae_auditedM"]), r["config"]) for r in qa])
for f_, nm in configs:
    axB.axvline(f_, color=MUTED, lw=0.8, alpha=0.5, zorder=0)
axB.text(0.985, 0.035, "grey lines: FMD of the 8 audited audited configurations",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=6.2,
         color=MUTED)

pairs = []
for c in [2.0, 5.0]:
    r2 = next(r for r in rows if r["base"] == BASE_CFG and F(r["c"]) == c
              and r["arm"] == "Q2_tail")
    r1 = next(r for r in rows if r["base"] == BASE_CFG and F(r["c"]) == c
              and r["bulk_law"] == LAW)
    pairs.append(dict(c=c, f1=F(r1["fmd_law"]), rho1=F(r1["rho_law"]),
                      f2=F(r2["fmd_law"]), rho2=F(r2["rho_law"]),
                      d=F(r1["delta_fmd_law"]),
                      dsd=F(r1["delta_fmd_sample_sd"])))
r0 = next(r for r in rows if r["base"] == BASE_CFG and r["arm"] == "Q0_base")
axB.scatter([F(r0["fmd_law"])], [F(r0["rho_law"])], s=46, marker="s",
            facecolor="white", edgecolor=INK, lw=1.1, zorder=5,
            label=r"$Q_0$ (EDM, unperturbed)")
for pr in pairs:
    col = BLUE if pr["c"] == 2 else ORANGE
    axB.plot([pr["f1"], pr["f2"]], [pr["rho1"], pr["rho2"]], color=col,
             lw=1.7, zorder=3, solid_capstyle="round")
    axB.scatter([pr["f1"]], [pr["rho1"]], s=60, marker="o", facecolor="white",
                edgecolor=col, lw=1.7, zorder=6)
    axB.scatter([pr["f2"]], [pr["rho2"]], s=60, marker="o", color=col, zorder=6)
    axB.annotate(rf"$c={pr['c']:g}$", (pr["f2"], pr["rho2"]),
                 textcoords="offset points", xytext=(9, -2), fontsize=8.5,
                 color=col, fontweight="bold")
axB.scatter([], [], s=60, marker="o", facecolor="white", edgecolor=INK, lw=1.7,
            label=r"$Q_1=(1{-}\varepsilon)Q_0+\varepsilon Q_B$  (matched bulk law)")
axB.scatter([], [], s=60, marker="o", color=INK,
            label=r"$Q_2=(1{-}\varepsilon)Q_0+\varepsilon Q_T$  (tail law)")
axB.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=1)
axB.text(0.985, 1.0, "calibrated", transform=axB.get_yaxis_transform(),
         ha="right", va="bottom", fontsize=7, color=MUTED)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlim(min(f for f, _ in configs) * 0.72, max(f for f, _ in configs) * 1.45)
axB.set_ylim(0.72, 12)
axB.set_xticks([0.3, 0.5, 1.0, 2.0, 4.0])
axB.set_xticklabels(["0.3", "0.5", "1", "2", "4"])
axB.set_yticks([1, 2, 5, 10]); axB.set_yticklabels(["1", "2", "5", "10"])
axB.set_xlabel("FMD (Fréchet distance, MAE ViT-B/16, audited $32^2$ sets)")
axB.set_ylabel(r"tail-calibration ratio  $\rho(10^{-3})$")
axB.set_title("One moment vector, two tail rates", fontweight="bold",
              loc="left")
axB.legend(loc="upper left", frameon=False, handletextpad=0.4, fontsize=6.9,
           borderpad=0.15, labelspacing=0.35)
axB.grid(alpha=0.18, lw=0.5, axis="y")

# ---------------- Panel C: how small is that FMD gap? ----------------------
gap_min = F(rows[0]["yard_min_adjacent_zoo_fmd_gap"])
gap_med = F(rows[0]["yard_median_adjacent_zoo_fmd_gap"])
floor = max(F(r["fmd_real_floor_sameM"]) for r in qa
            if int(r["M"]) == 100000)
p2, p5 = pairs[0], pairs[1]
def dof(law, c):
    r = next(r for r in rows if r["base"] == BASE_CFG and F(r["c"]) == c
             and r["bulk_law"] == law)
    return abs(F(r["delta_fmd_law"])), F(r["delta_fmd_sample_sd"])

items = [
    (rf"$|\Delta$FMD$|$, $c=2$, mean-matched", abs(p2["d"]), BLUE),
    (rf"$|\Delta$FMD$|$, $c=5$, mean-matched", abs(p5["d"]), ORANGE),
    (rf"$|\Delta$FMD$|$, $c=5$, 2nd-moment-matched arm",
     dof(LAW2, 5.0)[0], ORANGE),
    (rf"$|\Delta$FMD$|$, $c=5$, UNMATCHED control",
     dof("random", 5.0)[0], "#8f2358"),
    ("FMD Monte-Carlo s.d. at $M=10^5$", max(p5["dsd"], dof(LAW2, 5.0)[1]),
     MUTED),
    ("real-vs-real FMD floor at $M=10^5$", floor, MUTED),
    ("smallest FMD gap between two audited models", gap_min, INK),
    ("median FMD gap between adjacent audited models", gap_med, INK),
]
items = sorted(items, key=lambda t: t[1])
ys = np.arange(len(items))
for y, (lab, v, col) in zip(ys, items):
    axC.hlines(y, 1e-5, v, color=col, lw=1.4, alpha=0.75)
    axC.scatter([v], [y], s=42, color=col, zorder=4)
    axC.text(v * 1.25, y, f"{v:.2g}", va="center", fontsize=7, color=col)
axC.set_yticks(ys)
axC.set_yticklabels([lab for lab, _, _ in items], fontsize=6.4)
axC.set_xscale("log")
axC.set_xlim(min(v for _, v, _ in items) * 0.35,
             max(v for _, v, _ in items) * 9)
axC.set_ylim(-0.7, len(items) - 0.3)
axC.set_xlabel("FMD units (log scale)")
axC.set_title("The gap against every yardstick", fontweight="bold", loc="left")
axC.grid(alpha=0.18, lw=0.5, axis="x")
axC.tick_params(axis="y", length=0)

fig.tight_layout()
out = f"{FIGS}/fig_prop1_schematic.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for pr in pairs:
    print(f"  c={pr['c']:g}: rho {pr['rho1']:.4f} -> {pr['rho2']:.4f} "
          f"(diff {pr['rho2']-pr['rho1']:.4f}); FMD {pr['f1']:.6f} vs "
          f"{pr['f2']:.6f} (diff {pr['d']:+.6f}); paired MC s.d. at M=1e5 "
          f"{pr['dsd']:.6f}")
print(f"  yardsticks: min adjacent audited gap {gap_min:.5f}, median {gap_med:.5f},"
      f" real-vs-real floor at M=1e5 {floor:.5f}")
print(f"  panel-B Q_B ({LAW}) mean residual = "
      f"{F(con[LAW]['resid_mean_rel_to_tail_offset']):.3e} x the tail law's "
      f"own offset from the real mean, support {con[LAW]['n_support']} atoms")
for lab, v, _ in items:
    print(f"    ladder: {lab.replace(chr(92)+'n',' '):52s} {v:.6f}")
