#!/usr/bin/env python3
"""
Per-subject sweep across datasets. Resumable: results append to a CSV and
already-processed subjects are skipped, so an interrupted run picks up where it
stopped. Safe to kill and restart.

  python run_sweep.py --datasets BNCI2014_001 BNCI2014_004
  nohup python run_sweep.py --stage 1 > sweep.log 2>&1 &
  python run_sweep.py --analyse-only          # just re-run the statistics

Per subject it records:
  * the gamma sweep (gamma=0 is the matched control, i.e. uniform weights)
  * the published-vs-noweight mechanism check
  * weight entropy, max weight, MI range          -> H1
  * whether the top-weighted band is the most discriminative one -> H1-refined

H1-refined exists because IV-2a S5 and S8 have peaked weights but negative
gain. If their peak sits on the wrong band, H1 needs stating as "gain depends
on the weights departing from uniform *in the right direction*", which is both
more accurate and more useful.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import StratifiedKFold, cross_val_score

from awfbcsp import make_pipelines
from awfbcsp.config import (BANDS_FIXED, BANDS_PAPER, DEFAULTS, STAGE0, STAGE1,
                            band_edges, load_dataset)
from awfbcsp.features import FilterBankCSP, MIBandWeights
from awfbcsp.pipelines import AWFBCSP
from awfbcsp.stats import cliffs_delta, cliffs_magnitude

warnings.filterwarnings("ignore")
OUT = Path("results")
CSV = OUT / "sweep.csv"
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def band_discriminability(F, y, band_slices, seed=0):
    """MI of each band's CSP features with the labels -- the reference ranking."""
    from sklearn.feature_selection import mutual_info_classif
    return np.array([mutual_info_classif(F[:, sl], y, random_state=seed).mean()
                     for sl in band_slices])


def process_subject(par, ds, subject, tau, n_components, n_splits, seed,
                    descriptor="logvar_li"):
    X, y, _ = par.get_data(dataset=ds, subjects=[subject])
    y = pd.factorize(y)[0]
    if len(np.unique(y)) < 2:
        return None
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)

    fb = FilterBankCSP(n_components=n_components).fit(X, y)
    F = fb.transform(X)
    kw = dict(features=F, band_slices=fb.band_slices_) if descriptor == "csp" else {}
    mw = MIBandWeights(tau=tau, descriptor=descriptor).fit(X, y, **kw)
    w = np.ravel(mw.weights_)
    disc = band_discriminability(F, y, fb.band_slices_, seed)

    row = dict(dataset=ds.code, subject=int(subject), n_trials=int(len(y)),
               n_channels=int(X.shape[1]), n_bands=int(X.shape[3]), tau=tau,
               descriptor=descriptor,
               entropy=mw.weight_entropy_,
               deficit=mw.uniform_entropy_ - mw.weight_entropy_,
               max_w=float(w.max()), mi_range=float(np.ptp(np.ravel(mw.mi_))),
               argmax_w=int(np.argmax(w)), argmax_disc=int(np.argmax(disc)),
               peak_correct=int(np.argmax(w) == np.argmax(disc)),
               weights=";".join("%.4f" % v for v in w),
               disc=";".join("%.4f" % v for v in disc))

    for g in GAMMAS:
        pipes = make_pipelines(n_components=n_components, tau=tau, gamma=g)
        p = pipes["AWFBCSP-reg"]
        p.named_steps["feat"].set_params(descriptor=descriptor)
        row["g%s" % g] = float(cross_val_score(p, X, y, cv=cv).mean())
    row["gain"] = row["g1.0"] - row["g0.0"]
    row["gain_best"] = max(row["g%s" % g] for g in GAMMAS) - row["g0.0"]

    for name in ("FBCSP", "FBCSP+MIBIF", "AWFBCSP-published", "AWFBCSP-noweight"):
        try:
            p = make_pipelines(n_components=n_components, tau=tau)[name]
            if hasattr(p.named_steps.get("feat"), "set_params"):
                p.named_steps["feat"].set_params(descriptor=descriptor)
            row[name] = float(cross_val_score(p, X, y, cv=cv).mean())
        except Exception:
            row[name] = np.nan
    if not np.isnan(row.get("AWFBCSP-published", np.nan)):
        row["published_minus_noweight"] = abs(
            row["AWFBCSP-published"] - row["AWFBCSP-noweight"])
    return row


def analyse(df):
    L = ["=" * 72, "ANALYSIS  (n=%d subjects, %d datasets)"
         % (len(df), df["dataset"].nunique()), "=" * 72]

    L.append("\n[1] Mechanism: does the published sqrt(w) weighting do anything?")
    if "published_minus_noweight" in df:
        d = df["published_minus_noweight"].dropna()
        if len(d):
            L.append("    max |published - noweight| over subjects : %.3e" % d.max())
            L.append("    subjects with any difference            : %d/%d"
                     % ((d > 1e-9).sum(), len(d)))

    L.append("\n[2] Fixed gamma, mean over subjects (no per-subject tuning)")
    base = df["g0.0"].mean()
    for g in GAMMAS:
        c = "g%s" % g
        L.append("    gamma=%-5s %.4f   (vs gamma=0: %+.4f)"
                 % (g, df[c].mean(), df[c].mean() - base))

    L.append("\n[3] gamma=1.0 vs matched control gamma=0.0")
    a, b = df["g1.0"].to_numpy(), df["g0.0"].to_numpy()
    if len(a) >= 5 and not np.allclose(a, b):
        p = wilcoxon(a, b).pvalue
        cd = cliffs_delta(a, b)
        L += ["    mean diff       %+.4f  (%.2f pts)" % ((a - b).mean(), (a - b).mean() * 100),
              "    median diff     %+.4f" % np.median(a - b),
              "    improved        %d/%d" % ((a > b).sum(), len(a)),
              "    Wilcoxon p      %.4f" % p,
              "    Cliff's delta   %+.3f (%s)" % (cd, cliffs_magnitude(cd))]

    L.append("\n[4] H1: gain grows as the weights depart from uniform")
    for col in ("deficit", "max_w", "mi_range"):
        if col in df and df[col].notna().sum() >= 5:
            r, p = spearmanr(df[col], df["gain"])
            L.append("    Spearman(%-9s, gain)  rho=%+.3f  p=%.4f" % (col, r, p))

    L.append("\n[5] H1-refined: does the peak sit on the most discriminative band?")
    if "peak_correct" in df:
        ok = df["peak_correct"] == 1
        L.append("    peak correct    %d/%d subjects" % (ok.sum(), len(df)))
        if ok.sum() >= 3 and (~ok).sum() >= 3:
            L += ["    mean gain, peak correct   %+.4f (n=%d)"
                  % (df.loc[ok, "gain"].mean(), ok.sum()),
                  "    mean gain, peak wrong     %+.4f (n=%d)"
                  % (df.loc[~ok, "gain"].mean(), (~ok).sum())]
            from scipy.stats import mannwhitneyu
            L.append("    Mann-Whitney p            %.4f"
                     % mannwhitneyu(df.loc[ok, "gain"], df.loc[~ok, "gain"]).pvalue)
            sub = df[ok]
            if len(sub) >= 5:
                r, p = spearmanr(sub["deficit"], sub["gain"])
                L.append("    Spearman(deficit, gain) among peak-correct: "
                         "rho=%+.3f p=%.4f  (n=%d)" % (r, p, len(sub)))

    L.append("\n[6] Per dataset")
    for name, g in df.groupby("dataset"):
        L.append("    %-16s n=%-3d  gain=%+.4f  peak_correct=%d/%d"
                 % (name, len(g), (g["g1.0"] - g["g0.0"]).mean(),
                    g["peak_correct"].sum(), len(g)))
    L.append("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--n-components", type=int, default=DEFAULTS["n_components"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--bands", default="paper", choices=["paper", "fixed"],
                    help="paper: 8-30 Hz in five bands, as published, which "
                         "leaves 4-8 Hz outside every band and makes the last "
                         "band 6 Hz wide. fixed: 4-32 Hz in seven 4 Hz bands.")
    ap.add_argument("--descriptor", default="logvar_li",
                    choices=["logvar_li", "logpower", "csp"],
                    help="how band relevance is estimated; logvar_li is the "
                         "manuscript's aggregate descriptor")
    ap.add_argument("--out", default=None, help="override the results csv")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tag = args.descriptor + ("" if args.bands == "paper" else "_fixedbands")
    csv = Path(args.out) if args.out else (
        CSV if tag == "logvar_li" else OUT / ("sweep_%s.csv" % tag))
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
        print("\n=== %s  (%d subjects, descriptor=%s) ===" \
              % (ds.code, len(subs),
                 "%s/%s bands" % (args.descriptor, len(bands))), flush=True)
        for s in subs:
            if (ds.code, int(s)) in seen:
                print("  S%-4s cached" % s, flush=True)
                continue
            try:
                row = process_subject(par, ds, s, args.tau, args.n_components,
                                      args.n_splits, args.seed, args.descriptor)
            except Exception as e:
                print("  S%-4s FAILED %s: %s" % (s, type(e).__name__, e), flush=True)
                continue
            if row is None:
                continue
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                       index=False)
            print("  S%-4s gain=%+.4f  H=%.3f  max_w=%.3f  peak_%s"
                  % (s, row["gain"], row["entropy"], row["max_w"],
                     "ok" if row["peak_correct"] else "WRONG"), flush=True)

    if csv.exists():
        print("\n" + analyse(pd.read_csv(csv)))
        print("\nrows in %s" % csv.resolve())


if __name__ == "__main__":
    main()
