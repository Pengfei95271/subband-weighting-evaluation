#!/usr/bin/env python3
"""
Is the published example weight vector reachable under the released code?

Eq. (18) of the manuscript reports
    w = [0.0858, 0.0946, 0.6339, 0.0999, 0.0858]

The released implementation computes band weights as

    mi_scores = (mi - min(mi)) / (max(mi) - min(mi) + 1e-10)   # min-max to [0,1]
    weights   = softmax(mi_scores / 0.5)                       # tau = 0.5, hard-coded

Neither the min-max step nor tau = 0.5 appears in the paper. This script asks
three questions:

  A. What is the largest weight this formula can ever produce for B bands?
  B. What weights does it produce on real EEG, across many subjects?
  C. Is there any (tau, normalisation) combination that yields Eq. (18)?

Run:
  python verify_weights.py                 # A and C only, no data needed
  python verify_weights.py --with-data     # adds B, needs MOABB + IV-2a
"""

import argparse

import numpy as np

PUBLISHED = np.array([0.0858, 0.0946, 0.6339, 0.0999, 0.0858])
TAU_CODE = 0.5          # hard-coded in _compute_band_weights
EPS = 1e-10


def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + EPS)


def softmax(x, tau):
    z = np.asarray(x, float) / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def code_weights(mi, tau=TAU_CODE):
    """Exactly what the released code does."""
    return softmax(minmax(mi), tau)


# --------------------------------------------------------------------------
# A. analytic ceiling
# --------------------------------------------------------------------------

def ceiling(n_bands, tau=TAU_CODE):
    """Largest achievable weight: one band at 1, all others at 0."""
    best = np.zeros(n_bands)
    best[0] = 1.0
    return code_weights_from_normalised(best, tau).max()


def code_weights_from_normalised(s, tau):
    """Skip min-max (s is already normalised) and apply the softmax."""
    return softmax(s, tau)


def section_a():
    print("=" * 66)
    print("A. Analytic ceiling of the released formula")
    print("=" * 66)
    print("   min-max forces max(score)=1 and min(score)=0, so the most")
    print("   peaked weight vector possible is one band at 1, rest at 0.\n")
    print("   %-8s %-14s %-14s" % ("B", "ceiling", "uniform"))
    print("   " + "-" * 38)
    for b in (3, 5, 7, 9):
        print("   %-8d %-14.4f %-14.4f" % (b, ceiling(b), 1.0 / b))
    c5 = ceiling(5)
    print("\n   published max weight : %.4f" % PUBLISHED.max())
    print("   ceiling at B=5       : %.4f" % c5)
    print("   reachable            : %s" % (PUBLISHED.max() <= c5 + 1e-9))
    if PUBLISHED.max() <= c5:
        gap = c5 - PUBLISHED.max()
        print("   -> within %.4f of the ceiling; see section [c2] for how many"
              % gap)
        print("      bands must share the lowest raw MI to reach it.")
    return c5


# --------------------------------------------------------------------------
# C. what would it take?
# --------------------------------------------------------------------------

def section_c():
    print("\n" + "=" * 66)
    print("C. What settings would reproduce Eq. (18)?")
    print("=" * 66)

    # the published vector implies these normalised scores, up to a constant
    lg = np.log(PUBLISHED)
    lg = lg - lg.min()

    print("\n   [c1] With min-max normalisation, which tau gets closest?")
    print("        %-8s %-10s %-10s %-28s" % ("tau", "max_w", "L1 err", "weights"))
    print("        " + "-" * 62)
    best = None
    for tau in (2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        # the implied normalised scores that the published vector would need
        s = lg / (lg.max() + EPS)          # forced into [0,1] by min-max
        w = softmax(s, tau)
        err = np.abs(np.sort(w)[::-1] - np.sort(PUBLISHED)[::-1]).sum()
        flag = "  <- code default" if abs(tau - TAU_CODE) < 1e-9 else ""
        print("        %-8s %-10.4f %-10.4f %-28s%s"
              % (tau, w.max(), err, np.round(np.sort(w)[::-1], 3).tolist(), flag))
        if best is None or err < best[1]:
            best = (tau, err)
    print("\n        closest tau = %s (L1 error %.4f)" % best)

    print("\n   [c2] What raw MI spread would the code need at tau=0.5?")
    # solve: softmax(minmax(mi)/0.5) == PUBLISHED  =>  minmax(mi) = 0.5*(log w + c)
    s_needed = TAU_CODE * (np.log(PUBLISHED) - np.log(PUBLISHED).min())
    print("        implied normalised scores : %s"
          % np.round(s_needed, 4).tolist())
    print("        max of those              : %.4f" % s_needed.max())
    if s_needed.max() > 1.0 + 1e-9:
        print("        -> EXCEEDS 1.0, impossible after min-max normalisation")
    else:
        n_zero = int((s_needed < 1e-6).sum())
        print("        -> requires %d of %d bands to share the identical raw MI"
              % (n_zero, len(s_needed)))

    print("\n   [c3] Without the min-max step (paper's Eq. 8-9 as written):")
    for tau in (0.1, 0.05, 0.02, 0.01):
        mi_needed = tau * (np.log(PUBLISHED) - np.log(PUBLISHED).min())
        print("        tau=%-6s implied raw MI spread = %.4f nats  %s"
              % (tau, mi_needed.max(),
                 "(exceeds ln2=0.693, impossible for binary labels)"
                 if mi_needed.max() > np.log(2) else ""))


# --------------------------------------------------------------------------
# B. real data
# --------------------------------------------------------------------------

def band_descriptors_as_released(band_data):
    """Reproduce _compute_band_weights' feature list.

    band_data: (n_trials, n_channels, n_samples)
    Channel indices are hard-coded to the 22-channel BNCI2014-001 montage in
    the released code, and the laterality/suppression features are skipped
    entirely when fewer than 10 channels are present -- so IV-2b uses a
    different descriptor set from IV-2a.
    """
    feats = [np.mean(band_data ** 2, axis=(1, 2)),
             np.var(band_data, axis=(1, 2))]
    if band_data.shape[1] >= 10:
        left = np.mean(band_data[:, [6, 7, 8], :] ** 2, axis=(1, 2))
        right = np.mean(band_data[:, [10, 11, 12], :] ** 2, axis=(1, 2))
        feats.append((left - right) / (left + right + EPS))
    return np.column_stack(feats), band_data.shape[1] >= 10


def section_b(n_subjects, bands):
    from sklearn.feature_selection import mutual_info_classif
    from moabb.paradigms import FilterBankLeftRightImagery
    import pandas as pd
    from awfbcsp.config import band_edges, load_dataset

    print("\n" + "=" * 66)
    print("B. What the released formula produces on real EEG")
    print("=" * 66)

    par = FilterBankLeftRightImagery(filters=band_edges(bands), resample=250.0)
    ds = load_dataset("BNCI2014_001")
    rows = []
    for s in ds.subject_list[:n_subjects]:
        X, y, _ = par.get_data(dataset=ds, subjects=[s])
        y = pd.factorize(y)[0]
        mi = np.zeros(X.shape[3])
        for b in range(X.shape[3]):
            f, _ = band_descriptors_as_released(X[..., b])
            # the released code adds a Mu-suppression feature for bands 0-1
            if b < 2:
                f = np.column_stack([f, -np.mean(X[..., b] ** 2, axis=(1, 2))])
            mi[b] = mutual_info_classif(f, y, random_state=42).mean()
        w = code_weights(mi)
        rows.append(dict(subject=int(s), max_w=w.max(), mi_max=mi.max(),
                         mi_spread=float(mi.max() - mi.min()),
                         weights=np.round(w, 4).tolist()))
        print("   S%-3s max_w=%.4f  raw MI range=[%.4f, %.4f]  w=%s"
              % (s, w.max(), mi.min(), mi.max(), np.round(w, 3).tolist()))

    mx = np.array([r["max_w"] for r in rows])
    print("\n   across %d subjects: max weight  mean %.4f  range [%.4f, %.4f]"
          % (len(rows), mx.mean(), mx.min(), mx.max()))
    print("   published example max weight : %.4f" % PUBLISHED.max())
    print("   subjects reaching it         : %d/%d"
          % (int((mx >= PUBLISHED.max()).sum()), len(rows)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-data", action="store_true")
    ap.add_argument("--n-subjects", type=int, default=9)
    args = ap.parse_args()

    section_a()
    if args.with_data:
        from awfbcsp.config import BANDS_PAPER
        try:
            section_b(args.n_subjects, BANDS_PAPER)
        except Exception as e:
            print("\n   [B skipped: %s]" % e)
    section_c()
    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
