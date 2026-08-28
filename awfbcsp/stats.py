"""
Subject-level statistics and the Go/No-Go decision rule.

Everything here operates on a tidy DataFrame with columns
``subject``, ``pipeline``, ``score`` -- which is what MOABB evaluations return
(after averaging over sessions and folds).
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# Nemenyi critical values q_alpha at alpha = 0.05, infinite degrees of freedom
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def per_subject_scores(df, score_col="score"):
    """Collapse folds/sessions to one score per (pipeline, subject)."""
    key = ["dataset", "pipeline", "subject"] if "dataset" in df.columns \
        else ["pipeline", "subject"]
    return df.groupby(key, as_index=False)[score_col].mean()


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]. Positive means a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cliffs_magnitude(d):
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def paired_comparison(scores, method, reference):
    """Wilcoxon signed-rank of ``method`` against ``reference``, paired by subject."""
    piv = scores.pivot_table(index="subject", columns="pipeline", values="score")
    if method not in piv.columns or reference not in piv.columns:
        raise KeyError(f"missing pipeline: {method} or {reference}")
    piv = piv[[method, reference]].dropna()
    a, b = piv[method].to_numpy(), piv[reference].to_numpy()
    d = a - b
    if len(d) == 0:
        raise ValueError("no paired subjects")
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    return {
        "method": method,
        "reference": reference,
        "n_subjects": int(len(d)),
        "mean_method": float(a.mean()),
        "mean_reference": float(b.mean()),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "n_improved": int((d > 0).sum()),
        "p_value": p,
        "cliffs_delta": float(cliffs_delta(a, b)),
    }


def holm(pvalues):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def compare_all(scores, reference="FBCSP", alpha=0.05):
    """Every pipeline against the reference, Holm-corrected."""
    methods = [m for m in scores["pipeline"].unique() if m != reference]
    rows = [paired_comparison(scores, m, reference) for m in methods]
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].to_numpy())
    out["significant"] = out["p_holm"] < alpha
    out["effect"] = out["cliffs_delta"].map(cliffs_magnitude)
    return out.sort_values("mean_diff", ascending=False).reset_index(drop=True)


def friedman_nemenyi(scores, alpha=0.05):
    """Friedman omnibus test plus the Nemenyi critical difference."""
    piv = scores.pivot_table(index="subject", columns="pipeline",
                             values="score").dropna()
    k, n = piv.shape[1], piv.shape[0]
    if k < 3 or n < 3:
        return None
    stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    # rank 1 = best, so rank the negated scores
    ranks = piv.apply(lambda r: pd.Series(-r).rank(), axis=1)
    q = _Q05.get(k, _Q05[10])
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {
        "statistic": float(stat), "p_value": float(p),
        "n_subjects": int(n), "n_methods": int(k),
        "mean_ranks": ranks.mean().sort_values().to_dict(),
        "critical_difference": float(cd),
        "significant": bool(p < alpha),
    }


def cd_diagram(nemenyi, path="cd_diagram.png", title=None):
    """Critical-difference diagram. Methods within one CD are not distinguishable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if nemenyi is None:
        return None
    ranks = dict(sorted(nemenyi["mean_ranks"].items(), key=lambda kv: kv[1]))
    names, vals = list(ranks.keys()), list(ranks.values())
    k, cd = len(names), nemenyi["critical_difference"]
    lo, hi = np.floor(min(vals) - 0.5), np.ceil(max(vals) + 0.5)

    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.42 * k))
    ax.set_xlim(hi, lo)
    ax.set_ylim(0, k + 2.2)
    ax.axis("off")
    y0 = k + 1.2
    ax.plot([lo, hi], [y0, y0], color="black", lw=1)
    for t in np.arange(lo, hi + 0.001, 0.5):
        ax.plot([t, t], [y0, y0 + 0.12], color="black", lw=1)
        ax.text(t, y0 + 0.32, f"{t:g}", ha="center", va="bottom", fontsize=9)

    for i, (nm, v) in enumerate(zip(names, vals)):
        y = k - i
        ax.plot([v, v], [y0, y], color="0.4", lw=1)
        side = v < (lo + hi) / 2
        x_end = lo if side else hi
        ax.plot([v, x_end], [y, y], color="0.4", lw=1)
        ax.text(x_end, y, f" {nm} ({v:.2f}) " if side else f" {nm} ({v:.2f}) ",
                ha="left" if side else "right", va="center", fontsize=10)

    ax.plot([lo, lo + cd], [y0 + 0.9, y0 + 0.9], color="black", lw=2.5)
    ax.text(lo + cd / 2, y0 + 1.05, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# the decision rule
# --------------------------------------------------------------------------

GO = "GO"
CONDITIONAL = "CONDITIONAL"
NOGO = "NO-GO"


def verdict(scores, method="AWFBCSP-published", reference="FBCSP",
            diag=None, alpha=0.05, min_median_gain=0.01):
    """Apply the two-week Go/No-Go rule.

    NO-GO       p > 0.1 and median gain < 1 point -> stop adding experiments,
                go rebuild the method (plan section 2.2).
    CONDITIONAL significant but small, or mechanism shown to be inert.
    GO          significant, median gain at or above threshold, mechanism live.

    ``diag`` is the dict from diagnostics.weighting_is_effective, if available.
    """
    r = paired_comparison(scores, method, reference)
    gain, p = r["median_diff"], r["p_value"]
    mech_dead = diag is not None and not diag.get("effective", True)

    reasons = []
    if p > 0.1 and gain < min_median_gain:
        v = NOGO
        reasons.append(f"p={p:.3f} > 0.1 and median gain {gain*100:+.2f} pts "
                       f"< {min_median_gain*100:.1f}")
    elif p < alpha and gain >= min_median_gain:
        v = GO
        reasons.append(f"p={p:.4f} < {alpha}, median gain {gain*100:+.2f} pts")
    else:
        v = CONDITIONAL
        reasons.append(f"p={p:.3f}, median gain {gain*100:+.2f} pts")

    if mech_dead:
        reasons.append(
            "weighting has no numerical effect after standardisation "
            "(relative Frobenius diff "
            f"{diag.get('relative_frobenius_diff', float('nan')):.2e}) -- any "
            "observed difference comes from the interaction features or "
            "from seed variation, not from the weighting mechanism")
        if v == GO:
            v = CONDITIONAL

    r.update(verdict=v, reasons=reasons, mechanism_effective=not mech_dead)
    return r


def format_verdict(v):
    L = ["=" * 70, f"VERDICT: {v['verdict']}", "=" * 70,
         f"  {v['method']}  vs  {v['reference']}",
         f"  subjects            : {v['n_subjects']}",
         f"  mean score          : {v['mean_method']*100:.2f}%  vs  "
         f"{v['mean_reference']*100:.2f}%",
         f"  median gain         : {v['median_diff']*100:+.2f} points",
         f"  subjects improved   : {v['n_improved']}/{v['n_subjects']}",
         f"  Wilcoxon p          : {v['p_value']:.4f}",
         f"  Cliff's delta       : {v['cliffs_delta']:+.3f} "
         f"({cliffs_magnitude(v['cliffs_delta'])})",
         f"  mechanism effective : {v['mechanism_effective']}", ""]
    for r in v["reasons"]:
        L.append(f"  - {r}")
    L.append("=" * 70)
    return "\n".join(L)
