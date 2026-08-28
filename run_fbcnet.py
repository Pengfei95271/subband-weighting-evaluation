#!/usr/bin/env python3
"""
FBCNet against the CSP-family pipelines, same subjects, same folds.

FBCNet is the closest competitor: filter bank, spatial convolution, variance
layer -- the same processing chain, learned end to end instead of weighted.
It is also small, so it contests the lightweight-and-interpretable niche
directly. A reviewer will ask for it.

  python run_fbcnet.py --datasets BNCI2014_001 --max-subjects 2   # ~15 min
  nohup python run_fbcnet.py --stage 1 > fbcnet.log 2>&1 &
  python run_fbcnet.py --analyse-only

Resumable: rows append to results/fbcnet.csv, processed subjects skipped.
Hyperparameters are the published defaults and are held fixed across subjects
and folds; nothing is tuned per subject.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, cross_val_score

from awfbcsp import make_pipelines
from awfbcsp.config import BANDS_PAPER, DEFAULTS, STAGE0, STAGE1, band_edges, load_dataset
from awfbcsp.stats import cliffs_delta, cliffs_magnitude, holm

warnings.filterwarnings("ignore")
OUT = Path("results")
CSV = OUT / "fbcnet.csv"

CSP_PIPELINES = ["FBCSP", "FBCSP-L2", "AWFBCSP-published", "AWFBCSP-reg"]


def process_subject(par, ds, subject, n_splits, seed, tau, n_components,
                    max_epochs, m, patience=50):
    from awfbcsp.fbcnet import FBCNet

    X, y, _ = par.get_data(dataset=ds, subjects=[subject])
    y = pd.factorize(y)[0]
    if len(np.unique(y)) < 2:
        return None
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)

    row = dict(dataset=ds.code, subject=int(subject), n_trials=int(len(y)),
               n_channels=int(X.shape[1]), n_bands=int(X.shape[3]))

    t0 = time.time()
    fb = FBCNet(m=m, max_epochs=max_epochs, patience=patience,
                random_state=seed)
    row["FBCNet"] = float(cross_val_score(fb, X, y, cv=cv).mean())
    row["fbcnet_seconds"] = time.time() - t0
    fb.fit(X, y)
    row["fbcnet_params"] = int(fb.n_params_)

    pipes = make_pipelines(n_components=n_components, tau=tau,
                           random_state=seed)
    for name in CSP_PIPELINES:
        t0 = time.time()
        try:
            row[name] = float(cross_val_score(pipes[name], X, y, cv=cv).mean())
        except Exception:
            row[name] = np.nan
        row["%s_seconds" % name] = time.time() - t0
    return row


def analyse(df):
    L = ["=" * 70,
         "FBCNet vs CSP family  (n=%d subjects, %d datasets)"
         % (len(df), df["dataset"].nunique()), "=" * 70]

    methods = ["FBCNet"] + [m for m in CSP_PIPELINES if m in df.columns]

    L.append("\n[1] Mean accuracy across subjects")
    for m in methods:
        v = df[m].dropna()
        L.append("    %-20s %.4f  +/- %.4f  (n=%d)"
                 % (m, v.mean(), v.std(ddof=1), len(v)))

    L.append("\n[2] FBCNet against each CSP pipeline (Wilcoxon, Holm-corrected)")
    rows, ps = [], []
    for m in CSP_PIPELINES:
        if m not in df.columns:
            continue
        d = df[["FBCNet", m]].dropna()
        if len(d) < 6:
            continue
        a, b = d["FBCNet"].to_numpy(), d[m].to_numpy()
        p = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
        rows.append((m, (a - b).mean(), int((a > b).sum()), len(a), p,
                     cliffs_delta(a, b)))
        ps.append(p)
    if rows:
        adj = holm(ps)
        L.append("    %-20s %9s %9s %9s %9s %10s"
                 % ("vs", "diff", "better", "p", "p_holm", "delta"))
        L.append("    " + "-" * 70)
        for (m, dm, nb, n, p, cd), pa in zip(rows, adj):
            L.append("    %-20s %+8.4f %6d/%-3d %9.4f %9.4f %+7.3f (%s)"
                     % (m, dm, nb, n, p, pa, cd, cliffs_magnitude(cd)))

    L.append("\n[3] Cost")
    if "fbcnet_params" in df.columns:
        L.append("    FBCNet parameters      %d" % df["fbcnet_params"].median())
    for m in ["FBCNet"] + CSP_PIPELINES:
        c = "%s_seconds" % m.lower() if m == "FBCNet" else "%s_seconds" % m
        if c in df.columns:
            L.append("    %-20s %.1f s per subject (5-fold)"
                     % (m, df[c].mean()))

    L.append("\n[4] Per dataset")
    L.append("    %-16s %5s %10s %10s" % ("dataset", "n", "FBCNet", "FBCSP"))
    for name, g in df.groupby("dataset"):
        L.append("    %-16s %5d %10.4f %10.4f"
                 % (name, len(g), g["FBCNet"].mean(),
                    g["FBCSP"].mean() if "FBCSP" in g else np.nan))
    L.append("=" * 70)
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
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--m", type=int, default=32, help="FBCNet filters per band")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if args.analyse_only:
        if not len(done):
            sys.exit("no results yet")
        print(analyse(done))
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("FBCNet needs PyTorch.  pip install torch")

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
                row = process_subject(par, ds, s, args.n_splits, args.seed,
                                      args.tau, args.n_components,
                                      args.max_epochs, args.m, args.patience)
            except Exception as e:
                print("  S%-4s FAILED %s: %s" % (s, type(e).__name__, e),
                      flush=True)
                continue
            if row is None:
                continue
            pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(),
                                       index=False)
            print("  S%-4s FBCNet=%.4f  FBCSP=%.4f  (%.0fs)"
                  % (s, row["FBCNet"], row.get("FBCSP", np.nan),
                     row["fbcnet_seconds"]), flush=True)

    if CSV.exists():
        print("\n" + analyse(pd.read_csv(CSV)))


if __name__ == "__main__":
    main()
