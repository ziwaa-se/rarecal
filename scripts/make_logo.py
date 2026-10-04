"""Draw the RareCal logo (three concepts) as SVG + PNG.

  A  "tail lens"  : a density curve whose far tail is caught under a lens,
                    with the rare events shown as bright dots (default logo)
  B  "rho mark"   : a lowercase rho whose stem flows into a heavy tail
  C  "gauge"      : a calibration dial with the needle resting at 1

    python scripts/make_logo.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Wedge

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "logo")
os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, VIOLET, CORAL, AMBER, WHITE = "#0B1026", "#12B5A6", "#7B61FF", "#FF5E6C", "#FFB23F", "#F4F5FA"
plt.rcParams["font.family"] = "DejaVu Sans"


def grad_line(ax, x, y, c0, c1, lw, z=3, alpha=1.0):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", [c0, c1])
    lc = LineCollection(seg, colors=cmap(np.linspace(0, 1, len(seg))), lw=lw, capstyle="round",
                        zorder=z, alpha=alpha)
    ax.add_collection(lc)


def canvas(size=4, bg=NAVY, rounded=True):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if rounded:
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=bg, ec="none", zorder=0))
    return fig, ax


def icon_lens(ax, ox=0.0, oy=0.0, s=1.0, fg=WHITE):
    x = np.linspace(0.1, 0.9, 400)
    t = (x - 0.33) / 0.11
    y = 0.22 + 0.5 * (1 + t ** 2 / 3) ** (-2)                    # Student-t shaped density
    X, Y = ox + s * x, oy + s * y
    base = oy + s * 0.22
    ax.fill_between(X, base, Y, color=TEAL, alpha=0.10, lw=0, zorder=1)
    grad_line(ax, X, Y, TEAL, VIOLET, lw=5.5 * s, z=3)
    ax.plot([ox + s * 0.1, ox + s * 0.9], [base, base], color=fg, lw=2.2 * s, alpha=0.35,
            solid_capstyle="round", zorder=2)
    cx, cy, r = 0.69, 0.36, 0.14                                     # the lens over the tail
    ax.add_patch(Circle((ox + s * cx, oy + s * cy), s * r, fc=VIOLET, alpha=0.18, ec="none", zorder=4))
    ax.add_patch(Circle((ox + s * cx, oy + s * cy), s * r, fc="none", ec=fg, lw=4.2 * s, zorder=6))
    a = np.deg2rad(-45)
    x0, y0 = cx + r * np.cos(a), cy + r * np.sin(a)
    ax.plot([ox + s * x0, ox + s * (x0 + 0.11)], [oy + s * y0, oy + s * (y0 - 0.11)], color=fg,
            lw=7 * s, solid_capstyle="round", zorder=6)
    for (dx, dy, rr, c) in [(0.64, 0.30, 0.021, CORAL), (0.72, 0.29, 0.016, AMBER), (0.76, 0.34, 0.013, CORAL),
                            (0.67, 0.40, 0.011, AMBER)]:
        ax.add_patch(Circle((ox + s * dx, oy + s * dy), s * rr * 1.9, fc=c, alpha=0.25, ec="none", zorder=5))
        ax.add_patch(Circle((ox + s * dx, oy + s * dy), s * rr, fc=c, ec="none", zorder=5))


def icon_rho(ax, fg=WHITE):
    th = np.linspace(0, 2 * np.pi, 300)
    grad_line(ax, 0.5 + 0.17 * np.cos(th), 0.58 + 0.17 * np.sin(th), TEAL, VIOLET, lw=16)
    yy = np.linspace(0.58, 0.14, 100)
    xx = 0.33 + 0.30 * ((0.58 - yy) / 0.44) ** 3                      # stem bends into a tail
    grad_line(ax, xx, yy, VIOLET, CORAL, lw=16)
    for dx, c in [(0.72, CORAL), (0.80, AMBER), (0.86, CORAL)]:
        ax.add_patch(Circle((dx, 0.14), 0.022, fc=c, ec="none", zorder=5))


def icon_gauge(ax, fg=WHITE):
    for a0, a1, c in [(120, 180, AMBER), (60, 120, TEAL), (0, 60, CORAL)]:
        ax.add_patch(Wedge((0.5, 0.36), 0.36, a0, a1, width=0.09, fc=c, ec=NAVY, lw=3, zorder=2))
    ax.plot([0.5, 0.5], [0.36, 0.64], color=fg, lw=8, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((0.5, 0.36), 0.05, fc=fg, ec="none", zorder=5))
    ax.text(0.5, 0.18, "ρ = 1", ha="center", va="center", fontsize=30, color=fg, fontweight="bold")


def save(fig, name, transparent=False):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=128, transparent=True)
    plt.close(fig)


# individual icons
for name, fn in [("concept_A_lens", lambda ax: icon_lens(ax)), ("concept_B_rho", icon_rho),
                 ("concept_C_gauge", icon_gauge)]:
    fig, ax = canvas()
    fn(ax)
    save(fig, name)

# default logo = concept A
fig, ax = canvas()
icon_lens(ax)
save(fig, "logo")

# README / homepage banners: icon + wordmark, light and dark text
for mode, fg, sub in [("light", NAVY, "#4A5068"), ("dark", WHITE, "#B8BCCC")]:
    fig = plt.figure(figsize=(9, 2.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 9); ax.set_ylim(0, 2.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 1.9, 1.9, boxstyle="round,pad=0,rounding_size=0.38",
                                fc=NAVY, ec="none"))
    icon_lens(ax, ox=0.3, oy=0.3, s=1.6)
    t = ax.text(2.45, 1.28, "Rare", fontsize=54, fontweight="bold", color=fg, va="center")
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.02, 1.28, "Cal", fontsize=54, fontweight="bold", color=VIOLET, va="center")
    ax.text(2.5, 0.48, "audit & repair the rare events of generative models", fontsize=15.5,
            color=sub, va="center")
    save(fig, f"banner_{mode}")

# comparison sheet of the three concepts
fig = plt.figure(figsize=(12, 4.6))
for i, (fn, title) in enumerate([(lambda ax: icon_lens(ax), "A  tail lens (default)"),
                                 (icon_rho, "B  rho mark"), (icon_gauge, "C  calibration gauge")]):
    ax = fig.add_axes([0.03 + i * 0.33, 0.1, 0.27, 0.8]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=NAVY, ec="none", zorder=0))
    fn(ax)
    fig.text(0.03 + i * 0.33 + 0.135, 0.03, title, ha="center", fontsize=13, color=NAVY)
fig.savefig(os.path.join(OUT, "logo_concepts.png"), dpi=110, facecolor="white")
print("logo files ->", os.path.relpath(OUT))
