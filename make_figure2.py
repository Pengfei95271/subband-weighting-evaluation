#!/usr/bin/env python3
"""
Figure 2 -- does the band-relevance estimate identify the informative sub-band?

Reads results/descriptors.csv and writes figure2.pdf / figure2.png.

Panel (a): how often each descriptor's highest-weighted band is the band that
           actually classifies best, split by how well the subject decodes at
           all. The chance line is 1/n_bands.
Panel (b): rank agreement between the descriptor's mutual-information scores and
           the per-band accuracies, same split.

The split matters: an estimator that measures something real should improve as
the signal it is measuring gets stronger. One that does not, will not.

  python make_figure2.py
  python make_figure2.py --csv results/descriptors.csv --out figure2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

DESCRIPTORS = [
    ("logvar_li", "Aggregate descriptors\n(log power, variance, LI)", "#B4B2A9"),
    ("logpower", "Per-channel log power", "#5B9BD5"),
    ("csp", "CSP log-variance features", "#C1442A"),
]
TIERS = [(0.0, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
TIER_LABELS = ["<65%", "65-75%", "75-85%", ">85%"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
    })


def assign_tiers(df):
    t = pd.Series(index=df.index, dtype=object)
    for (lo, hi), lab in zip(TIERS, TIER_LABELS):
        t[(df["best_band_acc"] >= lo) & (df["best_band_acc"] < hi)] = lab
    return t


def build(df, out, chance=None):
    style()
    if chance is None:
        chance = 1.0 / float(df["n_bands"].median())
    df = df.copy()
    df["tier"] = assign_tiers(df)
    df = df[df["tier"].notna()]
    counts = [int((df["tier"] == t).sum()) for t in TIER_LABELS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(TIER_LABELS))
    w = 0.26

    # ---- panel a: correct rate ------------------------------------------
    for i, (key, label, colour) in enumerate(DESCRIPTORS):
        col = "%s_correct" % key
        if col not in df:
            continue
        rates, errs = [], [[], []]
        for t in TIER_LABELS:
            sub = df.loc[df["tier"] == t, col]
            k, n = int(sub.sum()), len(sub)
            r = k / n if n else np.nan
            rates.append(r)
            if n:
                lo, hi = binomtest(k, n, chance).proportion_ci(0.95)
                errs[0].append(max(r - lo, 0))
                errs[1].append(max(hi - r, 0))
            else:
                errs[0].append(0); errs[1].append(0)
        ax1.bar(x + (i - 1) * w, rates, w, label=label, color=colour,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax1.errorbar(x + (i - 1) * w, rates, yerr=errs, fmt="none",
                     ecolor="#3D3D3A", elinewidth=0.7, capsize=1.8, zorder=4)

    ax1.axhline(chance, color="#3D3D3A", lw=1.0, ls=(0, (4, 2)), zorder=5)
    ax1.text(len(TIER_LABELS) - 0.42, chance + 0.018,
             "chance (1/%d)" % round(1 / chance), fontsize=7,
             color="#3D3D3A", ha="right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["%s\n(n=%d)" % (t, c) for t, c in zip(TIER_LABELS, counts)])
    ax1.set_xlabel("Subject decoding level (best single-band accuracy)")
    ax1.set_ylabel("Informative band identified")
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.set_yticklabels(["%d%%" % (v * 100) for v in np.arange(0, 1.01, 0.2)])
    ax1.set_title("(a)", loc="left", fontweight="bold")
    ax1.grid(axis="y", lw=0.4, color="#E5E4DF", zorder=0)
    ax1.set_axisbelow(True)

    # ---- panel b: rank agreement ----------------------------------------
    for key, label, colour in DESCRIPTORS:
        col = "%s_rho" % key
        if col not in df:
            continue
        means = [df.loc[df["tier"] == t, col].mean() for t in TIER_LABELS]
        sems = [df.loc[df["tier"] == t, col].sem() for t in TIER_LABELS]
        ax2.errorbar(x, means, yerr=sems, marker="o", ms=4, lw=1.4,
                     color=colour, label=label, capsize=2, elinewidth=0.7)

    ax2.axhline(0, color="#3D3D3A", lw=1.0, ls=(0, (4, 2)))
    ax2.text(len(TIER_LABELS) - 1.05, 0.03, "no information", fontsize=7,
             color="#3D3D3A", ha="right")
    ax2.set_xticks(x)
    ax2.set_xticklabels(TIER_LABELS)
    ax2.set_xlabel("Subject decoding level")
    ax2.set_ylabel("Rank agreement with per-band accuracy\n(Spearman $\\rho$)")
    ax2.set_title("(b)", loc="left", fontweight="bold")
    ax2.grid(axis="y", lw=0.4, color="#E5E4DF")
    ax2.set_axisbelow(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ---- numbers for the caption ----------------------------------------
    lines = ["", "numbers for the caption:", "-" * 52]
    for key, label, _ in DESCRIPTORS:
        col = "%s_correct" % key
        if col not in df:
            continue
        k, n = int(df[col].sum()), len(df)
        p = binomtest(k, n, chance, alternative="greater").pvalue
        rho = df["%s_rho" % key].mean() if "%s_rho" % key in df else np.nan
        lines.append("  %-10s %3d/%-3d = %4.1f%%  p=%.2e  mean rho=%+.3f"
                     % (key, k, n, k / n * 100, p, rho))
    print("\n".join(lines))
    print("\nwrote %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/descriptors.csv")
    ap.add_argument("--out", default="figure2")
    ap.add_argument("--chance", type=float, default=None)
    args = ap.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit("not found: %s (run run_descriptors.py first)" % path)
    build(pd.read_csv(path), args.out, args.chance)


if __name__ == "__main__":
    main()
