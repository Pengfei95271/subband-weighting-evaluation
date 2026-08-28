#!/usr/bin/env python3
"""
Figure 3 -- does sub-band weighting improve decoding, once it can act at all?

Reads results/sweep.csv (aggregate descriptor) and results/sweep_csp.csv
(CSP-feature descriptor) and writes figure3.pdf / figure3.png.

Panel (a): change in accuracy against the matched control (gamma = 0, i.e.
           uniform weights, same pipeline, same classifier) as the weighting
           strength gamma increases. Two descriptors. Wilcoxon signed-rank
           against the control, marked where significant.
Panel (b): per-dataset mean change with 95% confidence intervals, sorted by
           cohort size, with the pooled estimate. This is the panel that shows
           why nine-subject evaluations produce the effect sizes reported in the
           literature: their intervals are wide enough to accommodate almost
           anything.

  python make_figure3.py
  python make_figure3.py --gamma 1.0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0]
SERIES = [
    ("results/sweep.csv", "Aggregate descriptors", "#B4B2A9", "o", "--"),
    ("results/sweep_csp.csv", "CSP log-variance features", "#C1442A", "s", "-"),
]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150,
    })


def ci95(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    h = student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - h, v.mean() + h


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def panel_a(ax, loaded):
    for label, colour, marker, ls, df in loaded:
        means, los, his, ps = [], [], [], []
        for g in GAMMAS:
            d = (df["g%s" % g] - df["g0.0"]).to_numpy() * 100
            means.append(d.mean())
            lo, hi = ci95(d)
            los.append(lo); his.append(hi)
            ps.append(1.0 if np.allclose(d, 0) else
                      wilcoxon(df["g%s" % g], df["g0.0"]).pvalue)
        x = np.arange(len(GAMMAS))
        ax.plot(x, means, marker=marker, ms=4.5, lw=1.5, ls=ls, color=colour,
                label="%s (n=%d)" % (label, len(df)), zorder=3)
        ax.fill_between(x, los, his, color=colour, alpha=0.16, lw=0, zorder=2)
        for xi, m, p in zip(x, means, ps):
            if stars(p):
                ax.annotate(stars(p), (xi, m), textcoords="offset points",
                            xytext=(0, -12 if m < 0 else 7), ha="center",
                            fontsize=8, color=colour)

    ax.axhline(0, color="#3D3D3A", lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.text(len(GAMMAS) - 1.02, 0.13, "matched control (uniform weights)",
            fontsize=7, color="#3D3D3A", ha="right")
    ax.set_xticks(np.arange(len(GAMMAS)))
    ax.set_xticklabels(["%g" % g for g in GAMMAS])
    ax.set_xlabel("Weighting strength $\\gamma$")
    ax.set_ylabel("Change in accuracy (percentage points)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.grid(axis="y", lw=0.4, color="#E5E4DF")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")


def panel_b(ax, df, gamma, colour="#C1442A"):
    rows = []
    for name, g in df.groupby("dataset"):
        d = (g["g%s" % gamma] - g["g0.0"]).to_numpy() * 100
        lo, hi = ci95(d)
        rows.append((name, len(g), d.mean(), lo, hi))
    rows.sort(key=lambda r: r[1])
    d_all = (df["g%s" % gamma] - df["g0.0"]).to_numpy() * 100
    lo, hi = ci95(d_all)
    rows.append(("Pooled", len(df), d_all.mean(), lo, hi))

    y = np.arange(len(rows))[::-1]
    for yi, (name, n, m, lo, hi) in zip(y, rows):
        pooled = name == "Pooled"
        c = "#3D3D3A" if pooled else colour
        ax.plot([lo, hi], [yi, yi], lw=1.4, color=c, zorder=3)
        ax.plot([m], [yi], marker="D" if pooled else "o",
                ms=6 if pooled else 4.5, color=c, zorder=4)
        ax.text(hi + 0.25, yi, "n=%d" % n, va="center", fontsize=7,
                color="#3D3D3A")

    ax.axhline(0.5, color="#E5E4DF", lw=0.8)
    ax.axvline(0, color="#3D3D3A", lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Change in accuracy at $\\gamma$=%g (percentage points)" % gamma)
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.grid(axis="x", lw=0.4, color="#E5E4DF")
    ax.set_axisbelow(True)
    xmax = max(r[4] for r in rows)
    ax.set_xlim(min(r[3] for r in rows) - 1.2, xmax + 2.2)
    return rows


def build(loaded, gamma, out):
    style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    panel_a(ax1, loaded)
    main = loaded[-1][-1]
    rows = panel_b(ax2, main, gamma)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), dpi=400, bbox_inches="tight")
    plt.close(fig)

    print("\nnumbers for the caption:\n" + "-" * 60)
    for label, _, _, _, df in loaded:
        print("  %s (n=%d)" % (label, len(df)))
        for g in GAMMAS[1:]:
            d = (df["g%s" % g] - df["g0.0"]) * 100
            p = wilcoxon(df["g%s" % g], df["g0.0"]).pvalue
            lo, hi = ci95(d)
            print("    gamma=%-4g %+.2f pts  [%+.2f, %+.2f]  p=%.4f  %s"
                  % (g, d.mean(), lo, hi, p, stars(p)))
    print("\n  per dataset at gamma=%g:" % gamma)
    for name, n, m, lo, hi in rows:
        print("    %-16s n=%-4d %+.2f  [%+.2f, %+.2f]  width=%.2f"
              % (name, n, m, lo, hi, hi - lo))
    print("\nwrote %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default="figure3")
    args = ap.parse_args()
    loaded = []
    for path, label, colour, marker, ls in SERIES:
        p = Path(path)
        if p.exists():
            loaded.append((label, colour, marker, ls, pd.read_csv(p)))
        else:
            print("missing %s -- skipping that series" % p)
    if not loaded:
        raise SystemExit("no sweep results found; run run_sweep.py first")
    build(loaded, args.gamma, args.out)


if __name__ == "__main__":
    main()
