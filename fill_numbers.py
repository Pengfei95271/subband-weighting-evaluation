#!/usr/bin/env python3
"""Pull the remaining [fill] values straight from the result files.

  python fill_numbers.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results")

def load(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


print("=" * 66)
print("Values still marked [fill] in the drafts")
print("=" * 66)

d = load("descriptors.csv")
if d is not None:
    print("\n[1] Dataset properties, as loaded")
    g = d.groupby("dataset").agg(subjects=("subject", "size"),
                                 channels=("n_channels", "median"),
                                 bands=("n_bands", "median"),
                                 trials=("n_trials", "median"),
                                 spread=("acc_spread", "mean"),
                                 best_acc=("best_band_acc", "mean"))
    g["channels"] = g["channels"].astype(int)
    g["trials"] = g["trials"].astype(int)
    print(g.round(4).to_string())
    print("\n    total subjects: %d" % len(d))
    print("    channel range : %d to %d" % (d["n_channels"].min(), d["n_channels"].max()))
    print("    mean spread   : %.2f pts" % (d["acc_spread"].mean() * 100))

f = load("fbcnet.csv")
if f is not None:
    print("\n[2] FBCNet")
    print("    median parameters : %d" % f["fbcnet_params"].median())
    print("    range             : %d to %d"
          % (f["fbcnet_params"].min(), f["fbcnet_params"].max()))
    print("    seconds/subject   : %.1f (5-fold)" % f["fbcnet_seconds"].mean())
    if "FBCSP_seconds" in f:
        print("    FBCSP seconds     : %.1f" % f["FBCSP_seconds"].mean())
        print("    ratio             : %.0fx" % (f["fbcnet_seconds"].mean()
                                                 / f["FBCSP_seconds"].mean()))

s = load("sweep_csp.csv")
if s is not None:
    print("\n[3] Training-fold size (5-fold, minus 20%% validation for FBCNet)")
    n = s["n_trials"]
    print("    median trials/subject : %d" % n.median())
    print("    per training fold     : %d" % int(n.median() * 0.8))
    print("    minus FBCNet val split: %d" % int(n.median() * 0.8 * 0.8))

print("\n[4] Per-subject-optimal gamma, as an upper bound")
if s is not None:
    gcols = [c for c in s.columns if c.startswith("g") and c != "g0.0"]
    best = s[gcols].max(axis=1) - s["g0.0"]
    fixed = s["g1.0"] - s["g0.0"]
    print("    best gamma per subject : %+.2f pts" % (best.mean() * 100))
    print("    fixed gamma = 1        : %+.2f pts" % (fixed.mean() * 100))
    print("    inflation              : %+.2f pts" % ((best - fixed).mean() * 100))

print("\n" + "=" * 66)
