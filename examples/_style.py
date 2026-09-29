"""Shared plotting style for the examples (matches the project logo)."""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, VIOLET, CORAL, AMBER, GREY = "#0B1026", "#12B5A6", "#7B61FF", "#FF5E6C", "#FFB23F", "#8A8FA3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C9CCD6",
    "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY, "text.color": NAVY,
    "axes.grid": True, "grid.color": "#EEF0F5", "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"figure -> {os.path.relpath(path)}")
