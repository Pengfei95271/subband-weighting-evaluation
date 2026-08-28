"""Offline check of the FBCNet baseline. No data download required.

  python test_fbcnet.py
"""

import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from awfbcsp.fbcnet import FBCNet
from test_synthetic import make_data

_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{'  [ok]  ' if cond else '  [FAIL]'} {name}{('  ' + detail) if detail else ''}")


def main():
    print("\n=== data ===")
    X, y, sb = make_data(n_trials=120, n_ch=12, n_times=500, seed=0)
    print(f"  X {X.shape}  (n_trials, n_ch, n_times, n_bands)  signal in band {sb}")

    print("\n=== 1. shape handling ===")
    Xt = FBCNet._to_nchw(X)
    check("transposed to (trials, bands, channels, times)",
          Xt.shape == (X.shape[0], X.shape[3], X.shape[1], X.shape[2]),
          str(Xt.shape))

    print("\n=== 2. fit and predict ===")
    clf = FBCNet(max_epochs=40, patience=10, random_state=0)
    clf.fit(X[:90], y[:90])
    pred = clf.predict(X[90:])
    check("predict returns one label per trial", len(pred) == 30)
    check("labels come from the training set",
          set(np.unique(pred)) <= set(np.unique(y)))
    proba = clf.predict_proba(X[90:])
    check("probabilities sum to 1", np.allclose(proba.sum(axis=1), 1.0))

    print("\n=== 3. parameter count ===")
    print(f"       {clf.n_params_:,} parameters")
    check("compact model (< 100k parameters)", clf.n_params_ < 100_000,
          f"{clf.n_params_:,}")

    print("\n=== 4. learns the synthetic signal ===")
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    acc = cross_val_score(FBCNet(max_epochs=60, patience=15, random_state=0),
                          X, y, cv=cv).mean()
    print(f"       3-fold accuracy {acc:.3f}")
    check("above chance", acc > 0.60, f"{acc:.3f}")

    print("\n=== 5. determinism ===")
    a = FBCNet(max_epochs=25, random_state=7).fit(X[:90], y[:90]).predict(X[90:])
    b = FBCNet(max_epochs=25, random_state=7).fit(X[:90], y[:90]).predict(X[90:])
    check("same seed gives same predictions", np.array_equal(a, b))

    print("\n=== 6. no leakage in normalisation ===")
    c = FBCNet(max_epochs=5, random_state=0).fit(X[:90], y[:90])
    mu_train = X[:90].mean(axis=(0, 1, 2))
    check("normalisation statistics come from training data only",
          np.allclose(np.ravel(c.mu_), mu_train, rtol=1e-4),
          "per-band mean matches the training split")

    n_pass, n_tot = sum(_results), len(_results)
    print(f"\n{'=' * 60}\n{n_pass}/{n_tot} checks passed\n{'=' * 60}")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
