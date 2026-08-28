#!/usr/bin/env python3
"""
Which descriptor identifies the informative sub-band?

The reference here is behavioural, not another MI estimate: for each sub-band we
fit CSP on that band alone and cross-validate, and the band with the highest
accuracy is taken as the one that actually carries the discriminative
information. Every descriptor is then scored on whether its highest-weighted
band is that band.

This matters because run_sweep.py's peak_correct compared one MI estimate
against another (band MI on CSP features), which cannot score the 'csp'
descriptor fairly -- it would be comparing that descriptor with itself.

  python run_descriptors.py --datasets BNCI2014_001 --max-subjects 3
  nohup python run_descriptors.py --stage 1 > descr.log 2>&1 &
  python run_descriptors.py --analyse-only

Resumable: rows append to results/descriptors.csv, processed subjects skipped.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from awfbcsp.config import (BANDS_FIXED, BANDS_PAPER, DEFAULTS, STAGE0, STAGE1,
                            band_edges, load_dataset)
from awfbcsp.features import CSP, FilterBankCSP, MIBandWeights

warnings.filterwarnings("ignore")
OUT = Path("results")
CSV = OUT / "descriptors.csv"
DESCRIPTORS = ("logvar_li", "logpower", "csp")


def per_band_accuracy(X, y, n_components, cv):
    """Behavioural reference: cross-validated accuracy of each band on its own."""
    acc = []
    for b in range(X.shape[3]):
        pipe = Pipeline([("csp", CSP(n_components=n_components)),
                         ("sc", StandardScaler()),
                         ("lda", LDA(solver="lsqr", shrinkage="auto"))])
        acc.append(float(cross_val_score(pipe, X[..., b], y, cv=cv).mean()))
    return np.array(acc)


def process_subject(par, ds, subject, tau, n_components, n_splits, seed):
    X, y, _ = par.get_data(dataset=ds, subjects=[subject])
    y = pd.factorize(y)[0]
    if len(np.unique(y)) < 2:
        return None
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)

    acc = per_band_accuracy(X, y, n_components, cv)
    best = int(np.argmax(acc))
    n_bands = X.shape[3]

    fb = FilterBankCSP(n_components=n_components).fit(X, y)
    F = fb.transform(X)

    row = dict(dataset=ds.code, subject=int(subject), n_bands=n_bands,
               n_channels=int(X.shape[1]), n_trials=int(len(y)),
               best_band=best, best_band_acc=float(acc[best]),
               acc_spread=float(acc.max() - acc.min()),
               band_acc=";".join("%.4f" % v for v in acc))

    for desc in DESCRIPTORS:
        kw = dict(features=F, band_slices=fb.band_slices_) if desc == "csp" else {}
        mw = MIBandWeights(tau=tau, descriptor=desc).fit(X, y, **kw)
        w = np.ravel(mw.weights_)
        mi = np.ravel(mw.mi_)
        rho = spearmanr(mi, acc).correlation if n_bands >= 3 else np.nan
        row["%s_argmax" % desc] = int(np.argmax(w))
        row["%s_correct" % desc] = int(np.argmax(w) == best)
        row["%s_rho" % desc] = float(rho) if np.isfinite(rho) else np.nan
        row["%s_maxw" % desc] = float(w.max())
        row["%s_entropy" % desc] = float(mw.weight_entropy_)
        row["%s_mi_max" % desc] = float(mi.max())
        # accuracy cost of following this descriptor instead of the best band
        row["%s_acc_cost" % desc] = float(acc[best] - acc[int(np.argmax(w))])
    return row


def analyse(df):
    L = ["=" * 72,
         "DESCRIPTOR COMPARISON  (n=%d subjects, %d datasets)"
         % (len(df), df["dataset"].nunique()),
         "reference: the sub-band that classifies best on its own", "=" * 72]

    nb = int(df["n_bands"].median())
    chance = 1.0 / nb
    L.append("\n[1] Does the descriptor pick the band that actually classifies best?")
    L.append("    chance level = 1/%d = %.1f%%\n" % (nb, chance * 100))
    L.append("    %-12s %10s %8s %10s %12s" %
             ("descriptor", "correct", "rate", "binom p", "vs chance"))
    L.append("    " + "-" * 56)
    for d in DESCRIPTORS:
        c = "%s_correct" % d
        if c not in df:
            continue
        k, n = int(df[c].sum()), len(df)
        p = binomtest(k, n, chance, alternative="greater").pvalue
        L.append("    %-12s %6d/%-4d %7.1f%% %10.4f %+11.1f pts"
                 % (d, k, n, k / n * 100, p, (k / n - chance) * 100))

    L.append("\n[2] Rank agreement between the descriptor's MI and per-band accuracy")
    L.append("    (Spearman per subject, averaged; 0 means no information)")
    for d in DESCRIPTORS:
        c = "%s_rho" % d
        if c in df and df[c].notna().sum():
            v = df[c].dropna()
            from scipy.stats import wilcoxon
            p = wilcoxon(v).pvalue if len(v) >= 6 and not np.allclose(v, 0) else np.nan
            L.append("    %-12s mean rho=%+.3f  median=%+.3f  p=%.4f  (n=%d)"
                     % (d, v.mean(), v.median(), p, len(v)))

    L.append("\n[3] Accuracy left on the table by following the descriptor")
    for d in DESCRIPTORS:
        c = "%s_acc_cost" % d
        if c in df:
            L.append("    %-12s mean cost %.4f  (%.2f pts below the best band)"
                     % (d, df[c].mean(), df[c].mean() * 100))
    if "acc_spread" in df:
        L.append("    %-12s mean spread across bands %.2f pts"
                 % ("(context)", df["acc_spread"].mean() * 100))

    L.append("\n[4] Per dataset, correct-rate by descriptor")
    L.append("    %-16s %5s %10s %10s %10s"
             % ("dataset", "n", *DESCRIPTORS))
    for name, g in df.groupby("dataset"):
        L.append("    %-16s %5d %9.0f%% %9.0f%% %9.0f%%"
                 % (name, len(g), *[g["%s_correct" % d].mean() * 100
                                    for d in DESCRIPTORS]))
    L.append("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--stage", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--n-components", type=int, default=DEFAULTS["n_components"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--bands", default="paper", choices=["paper", "fixed"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    csv = Path(args.out) if args.out else (
        CSV if args.bands == "paper" else OUT / "descriptors_fixedbands.csv")
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    if args.analyse_only:
        if not len(done):
            sys.exit("no results in %s" % csv)
        print(analyse(done))
        return

    names = args.datasets or (STAGE0 if args.stage == 0 else STAGE0 + STAGE1)
    from moabb.paradigms import FilterBankLeftRightImagery
    bands = BANDS_PAPER if args.bands == "paper" else BANDS_FIXED
    par = FilterBankLeftRightImagery(filters=band_edges(bands),
                                     resample=DEFAULTS["resample"])
    seen = set(zip(done["dataset"], done["subject"])) if len(done) else set()

    for name in names:
        try:
            ds = load_dataset(name)
        except Exception as e:
            print("skip %s (%s)" % (name, e), flush=True)
            continue
        subs = ds.subject_list[:args.max_subjects] if args.max_subjects \
            else ds.subject_list
        print("\n=== %s  (%d subjects, %d bands) ==="
              % (ds.code, len(subs), len(bands)), flush=True)
        for s in subs:
            if (ds.code, int(s)) in seen:
                print("  S%-4s cached" % s, flush=True)
                continue
            try:
                row = process_subject(par, ds, s, args.tau, args.n_components,
                                      args.n_splits, args.seed)
            except Exception as e:
                print("  S%-4s FAILED %s: %s" % (s, type(e).__name__, e), flush=True)
                continue
            if row is None:
                continue
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                       index=False)
            print("  S%-4s best_band=%d  %s"
                  % (s, row["best_band"],
                     "  ".join("%s=%s" % (d[:7], "ok " if row["%s_correct" % d]
                                          else "no ") for d in DESCRIPTORS)),
                  flush=True)

    if csv.exists():
        print("\n" + analyse(pd.read_csv(csv)))
        print("\nrows in %s" % csv.resolve())


if __name__ == "__main__":
    main()
