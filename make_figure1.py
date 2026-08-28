#!/usr/bin/env python3
"""
Figure 1 -- where band weighting enters, and why the usual order makes it inert.

A schematic; no data required.

  python make_figure1.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

GREY = "#8A8880"
GREY_FILL = "#EDECE8"
RED = "#C1442A"
RED_FILL = "#FBEDE9"
TEAL = "#1D7A6B"
TEAL_FILL = "#E8F3F0"
INK = "#2B2B28"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "figure.dpi": 150,
    })


def box(ax, x, y, w, h, text, fc, ec, fontsize=8, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.0,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=3, weight=weight,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, colour=INK, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw, color=colour,
                                 shrinkA=0, shrinkB=0, zorder=3))


def build(out):
    style()
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    bw, bh = 0.163, 0.155          # box width / height
    gap = 0.043
    x0 = 0.048
    xs = [x0 + i * (bw + gap) for i in range(4)]
    y_top, y_bot = 0.660, 0.245

    # ---------------- row a: the published order ----------------------
    ax.text(0.012, y_top + bh + 0.10, "(a)", fontsize=9, weight="bold", color=INK)
    ax.text(0.048, y_top + bh + 0.10,
            "As implemented: rescale, then standardise", fontsize=8.5,
            weight="bold", color=INK)

    box(ax, xs[0], y_top, bw, bh,
        "Filter-bank CSP\nlog-variance\nfeatures  $F^{(b)}$", "#FFFFFF", GREY)
    box(ax, xs[1], y_top, bw, bh,
        "Rescale band $b$\n$\\tilde F^{(b)}=\\sqrt{w_b}\\,F^{(b)}$",
        GREY_FILL, GREY)
    box(ax, xs[2], y_top, bw, bh,
        "Per-feature\n$z$-score", GREY_FILL, GREY)
    box(ax, xs[3], y_top, bw, bh, "Classifier", "#FFFFFF", GREY)
    for i in range(3):
        arrow(ax, xs[i] + bw, y_top + bh / 2, xs[i + 1], y_top + bh / 2, GREY)

    # the cancellation, drawn under the two boxes it concerns
    ax.plot([xs[1] + 0.012, xs[2] + bw - 0.012], [y_top - 0.045] * 2,
            color=GREY, lw=1.0)
    for xx in (xs[1] + 0.012, xs[2] + bw - 0.012):
        ax.plot([xx, xx], [y_top - 0.045, y_top - 0.028], color=GREY, lw=1.0)
    ax.text((xs[1] + xs[2] + bw) / 2, y_top - 0.088,
            "$z(c\\,f)=z(f)$  for any $c>0$   --   the factor is removed exactly",
            ha="center", fontsize=8, color=GREY)

    ax.text(xs[3] + bw + 0.022, y_top + bh / 2,
            "identical predictions\n211/211 subjects\n"
            "$\\Vert\\Delta F\\Vert_F/\\Vert F\\Vert_F \\approx 10^{-15}$",
            fontsize=7.6, color=GREY, va="center", linespacing=1.5)

    # ---------------- row b: the corrected order ----------------------
    ax.text(0.012, y_bot + bh + 0.10, "(b)", fontsize=9, weight="bold", color=INK)
    ax.text(0.048, y_bot + bh + 0.10,
            "Reformulated: standardise, then weight, with a penalised estimator",
            fontsize=8.5, weight="bold", color=INK)

    box(ax, xs[0], y_bot, bw, bh,
        "Filter-bank CSP\nlog-variance\nfeatures  $F^{(b)}$", "#FFFFFF", TEAL)
    box(ax, xs[1], y_bot, bw, bh, "Per-feature\n$z$-score", TEAL_FILL, TEAL)
    box(ax, xs[2], y_bot, bw, bh,
        "Rescale band $b$\n$w_b^{\\gamma/2}F^{(b)}$", TEAL_FILL, TEAL)
    box(ax, xs[3], y_bot, bw, bh,
        "$L_2$-penalised\nclassifier", "#FFFFFF", TEAL)
    for i in range(3):
        arrow(ax, xs[i] + bw, y_bot + bh / 2, xs[i + 1], y_bot + bh / 2, TEAL)

    ax.plot([xs[2] + 0.012, xs[3] + bw - 0.012], [y_bot - 0.045] * 2,
            color=TEAL, lw=1.0)
    for xx in (xs[2] + 0.012, xs[3] + bw - 0.012):
        ax.plot([xx, xx], [y_bot - 0.045, y_bot - 0.028], color=TEAL, lw=1.0)
    ax.text((xs[2] + xs[3] + bw) / 2, y_bot - 0.092,
            "solves $\\;\\min_\\beta\\, L(y,F\\beta)+\\lambda\\sum_b w_b^{-\\gamma}"
            "\\Vert\\beta_b\\Vert_2^2$",
            ha="center", fontsize=8, color=TEAL)

    ax.text(xs[3] + bw + 0.022, y_bot + bh / 2,
            "the solution changes\n"
            "both conditions are\nload-bearing",
            fontsize=7.6, color=TEAL, va="center", linespacing=1.5)

    # ---------------- the second, independent reason ------------------
    ax.add_patch(FancyBboxPatch((0.048, 0.008), 0.928, 0.080,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=RED_FILL, edgecolor=RED, linewidth=1.0,
                                zorder=2))
    ax.text(0.062, 0.048,
            "Independently of (a): six of the nine classifiers commonly paired "
            "with CSP features -- LDA, naive Bayes, decision tree,\nrandom "
            "forest, gradient boosting, AdaBoost -- are invariant to positive "
            "per-feature rescaling by construction.",
            fontsize=7.4, color=INK, va="center", ha="left", linespacing=1.55,
            zorder=3)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s.pdf and %s.png" % (out, out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figure1")
    build(ap.parse_args().out)
