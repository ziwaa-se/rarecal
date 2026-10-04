"""Rank correlations between quality axes and tail calibration, with leave-one-out sensitivity."""
import csv
import math
import os

from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SRC = os.path.join(DATA, "quality_axis_fmd.csv")
OUT = os.path.join(DATA, "quality_axis_matched_correlations.csv")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def approx_p(rho, n):
    """Two-sided p from the t distribution with n-2 df.

 Must be Student's t, not a normal approximation: at n=7 the normal tail
 understates p badly (0.039 vs the correct 0.094 for rho_s=+0.679), which
 would put the correlation on the wrong side of 0.05.
 """
    if n <= 2 or abs(rho) >= 1.0:
        return 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return float(2.0 * stats.t.sf(abs(t), df=n - 2))


rows = list(csv.DictReader(open(SRC)))
F = lambda r, k: float(r[k])
matched = [r for r in rows if r["fid_published"] not in ("", "nan", None)]

y_m = [F(r, "abs_log_rho") for r in matched]
fid_m = [F(r, "fid_published") for r in matched]
fmd_m = [F(r, "fmd_mae_auditedM") for r in matched]
y_a = [F(r, "abs_log_rho") for r in rows]
fmdc_a = [F(r, "fmd_mae_commonM") for r in rows]

out = []
for tag, xs, ys, n in [
        ("matched published FID vs |log rho|", fid_m, y_m, len(matched)),
        ("matched measured FMD vs |log rho|", fmd_m, y_m, len(matched)),
        ("all common-M FMD vs |log rho|", fmdc_a, y_a, len(rows))]:
    r = spearman(xs, ys)
    out.append(dict(tag=tag, n=n, spearman=round(r, 4),
                    spearman_p_approx=round(approx_p(r, n), 4),
                    dropped_config="", note="matched set = configs with both axes"))

for label, xs in [("published", fid_m), ("measured", fmd_m)]:
    loo = []
    for i in range(len(matched)):
        idx = [j for j in range(len(matched)) if j != i]
        r = spearman([xs[j] for j in idx], [y_m[j] for j in idx])
        loo.append((matched[i]["config"], r))
        out.append(dict(tag=f"LOO {label} FID/FMD vs |log rho|", n=len(idx),
                        spearman=round(r, 4),
                        spearman_p_approx=round(approx_p(r, len(idx)), 4),
                        dropped_config=matched[i]["config"], note="leave-one-out"))
    lo, hi = min(r for _, r in loo), max(r for _, r in loo)
    out.append(dict(tag=f"LOO range {label}", n=len(matched) - 1,
                    spearman=f"[{lo:+.2f},{hi:+.2f}]", spearman_p_approx="",
                    dropped_config="", note="range over all single deletions"))
    print(f"{label:10s} LOO range [{lo:+.2f}, {hi:+.2f}]", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

for r in out[:3]:
    print(f"{r['tag']:38s} n={r['n']}  rho_s={r['spearman']:+}  "
          f"p~{r['spearman_p_approx']}", flush=True)
print(f"wrote {os.path.basename(OUT)} ({len(out)} rows)", flush=True)
