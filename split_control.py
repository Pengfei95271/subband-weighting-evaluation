#!/usr/bin/env python3
"""
Split-half control for the relevance-quality effect.

The concern this addresses: csp_rho and the weighting gain are computed from the
same trials, so a band that is favourable by sampling accident could inflate
both, producing a correlation with no underlying band structure. That would be
enough to overturn the finding, and a reviewer will raise it.

Here the trials are split in half, stratified by class:
  half A -> estimate band relevance, and the behavioural reference
  half B -> measure the weighting gain

Any correlation that survives cannot come from shared sampling noise.

  python split_control.py --stage 1
  python split_control.py --datasets BNCI2014_001 --max-subjects 3
  python split_control.py --analyse-only

Resumable: rows append to results/split_control.csv.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from awfbcsp import make_pipelines
from awfbcsp.config import BANDS_PAPER, DEFAULTS, STAGE0, STAGE1, band_edges, load_dataset
from awfbcsp.features import CSP, FilterBankCSP, MIBandWeights

warnings.filterwarnings("ignore")
OUT = Path("results")
CSV = OUT / "split_control.csv"


def band_accuracy(X, y, n_components, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = lambda: Pipeline([("csp", CSP(n_components=n_components)),
                             ("sc", StandardScaler()),
                             ("lda", LDA(solver="lsqr", shrinkage="auto"))])
    return np.array([cross_val_score(pipe(), X[..., b], y, cv=cv).mean()
                     for b in range(X.shape[3])])


def process_subject(par, ds, subject, tau, n_components, seed):
    X, y, _ = par.get_data(dataset=ds, subjects=[subject])
    y = pd.factorize(y)[0]
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 20:
        return None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    iA, iB = next(sss.split(np.zeros(len(y)), y))
    XA, yA, XB, yB = X[iA], y[iA], X[iB], y[iB]

    # ---- half A: estimate relevance and the behavioural reference ----
    fbA = FilterBankCSP(n_components=n_components).fit(XA, yA)
    FA = fbA.transform(XA)
    mwA = MIBandWeights(tau=tau, descriptor="csp").fit(
        XA, yA, features=FA, band_slices=fbA.band_slices_)
    accA = band_accuracy(XA, yA, n_components, seed)
    rhoA = spearmanr(np.ravel(mwA.mi_), accA).correlation
    okA = int(np.argmax(np.ravel(mwA.weights_)) == int(np.argmax(accA)))

    # ---- half B: measure the gain, independently ----
    cvB = StratifiedKFold(5, shuffle=True, random_state=seed)
    g = {}
    for gam in (0.0, 0.5, 1.0):
        p = make_pipelines(n_components=n_components, tau=tau, gamma=gam)["AWFBCSP-reg"]
        p.named_steps["feat"].set_params(descriptor="csp")
        g[gam] = float(cross_val_score(p, XB, yB, cv=cvB).mean())

    # ---- same-half version, to quantify how much of the effect was shared noise ----
    cvA = StratifiedKFold(5, shuffle=True, random_state=seed)
    gA = {}
    for gam in (0.0, 1.0):
        p = make_pipelines(n_components=n_components, tau=tau, gamma=gam)["AWFBCSP-reg"]
        p.named_steps["feat"].set_params(descriptor="csp")
        gA[gam] = float(cross_val_score(p, XA, yA, cv=cvA).mean())

    return dict(dataset=ds.code, subject=int(subject), n_trials=int(len(y)),
                rhoA=float(rhoA) if np.isfinite(rhoA) else np.nan, okA=okA,
                best_band_A=int(np.argmax(accA)), best_acc_A=float(accA.max()),
                gainB_g05=(g[0.5] - g[0.0]) * 100,
                gainB=(g[1.0] - g[0.0]) * 100,
                gainA_same_half=(gA[1.0] - gA[0.0]) * 100)


def analyse(df):
    L = ["=" * 68,
         "SPLIT-HALF CONTROL  (n=%d subjects, %d datasets)" % (len(df),
                                                               df["dataset"].nunique()),
         "relevance estimated on half A, gain measured on half B", "=" * 68]

    L.append("\n[1] Does relevance quality on half A predict the gain on half B?")
    for col, lab in (("rhoA", "rank agreement"), ("okA", "peak correct")):
        v = df[[col, "gainB"]].dropna()
        if len(v) >= 10:
            r, p = spearmanr(v[col], v["gainB"])
            L.append("    Spearman(%-14s, gainB)  rho=%+.3f  p=%.4f  (n=%d)"
                     % (lab, r, p, len(v)))

    L.append("\n[2] Split by peak correctness on half A")
    ok = df["okA"] == 1
    if ok.sum() >= 5 and (~ok).sum() >= 5:
        L += ["    peak correct  n=%3d   gain on half B  %+.2f pts"
              % (ok.sum(), df.loc[ok, "gainB"].mean()),
              "    peak wrong    n=%3d   gain on half B  %+.2f pts"
              % ((~ok).sum(), df.loc[~ok, "gainB"].mean()),
              "    Mann-Whitney p = %.4f"
              % mannwhitneyu(df.loc[ok, "gainB"], df.loc[~ok, "gainB"]).pvalue,
              "    separation     %+.2f pts"
              % (df.loc[ok, "gainB"].mean() - df.loc[~ok, "gainB"].mean())]

    L.append("\n[3] How much of the original effect was shared sampling noise?")
    if "gainA_same_half" in df:
        for col, lab in (("gainA_same_half", "same half (confounded)"),
                         ("gainB", "held-out half (clean)")):
            v = df[["okA", col]].dropna()
            o = v["okA"] == 1
            if o.sum() >= 5 and (~o).sum() >= 5:
                L.append("    %-24s correct %+.2f   wrong %+.2f   separation %+.2f"
                         % (lab, v.loc[o, col].mean(), v.loc[~o, col].mean(),
                            v.loc[o, col].mean() - v.loc[~o, col].mean()))
        L.append("    (the original analysis used the confounded version;"
                 " the clean one is the test)")

    L.append("\n[4] Pooled gain on the held-out half")
    for col, lab in (("gainB_g05", "gamma=0.5"), ("gainB", "gamma=1.0")):
        if col in df:
            from scipy.stats import wilcoxon
            v = df[col].dropna()
            p = wilcoxon(v).pvalue if len(v) >= 6 and not np.allclose(v, 0) else np.nan
            L.append("    %-10s  %+.2f pts   p=%.4f" % (lab, v.mean(), p))

    L.append("\n[5] Per dataset")
    for name, g in df.groupby("dataset"):
        L.append("    %-16s n=%-4d rhoA=%+.3f  gainB=%+.2f  peak_ok=%d/%d"
                 % (name, len(g), g["rhoA"].mean(), g["gainB"].mean(),
                    int(g["okA"].sum()), len(g)))
    L.append("=" * 68)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--stage", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--n-components", type=int, default=DEFAULTS["n_components"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if args.analyse_only:
        if not len(done):
            sys.exit("no results yet")
        print(analyse(done))
        return

    names = args.datasets or (STAGE0 if args.stage == 0 else STAGE0 + STAGE1)
    from moabb.paradigms import FilterBankLeftRightImagery
    par = FilterBankLeftRightImagery(filters=band_edges(BANDS_PAPER),
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
        print("\n=== %s  (%d subjects) ===" % (ds.code, len(subs)), flush=True)
        for s in subs:
            if (ds.code, int(s)) in seen:
                print("  S%-4s cached" % s, flush=True)
                continue
            try:
                row = process_subject(par, ds, s, args.tau, args.n_components,
                                      args.seed)
            except Exception as e:
                print("  S%-4s FAILED %s: %s" % (s, type(e).__name__, e), flush=True)
                continue
            if row is None:
                print("  S%-4s skipped (too few trials)" % s, flush=True)
                continue
            pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(),
                                       index=False)
            print("  S%-4s rhoA=%+.3f  peak_%s  gainB=%+.2f"
                  % (s, row["rhoA"], "ok " if row["okA"] else "no ", row["gainB"]),
                  flush=True)

    if CSV.exists():
        print("\n" + analyse(pd.read_csv(CSV)))


if __name__ == "__main__":
    main()
