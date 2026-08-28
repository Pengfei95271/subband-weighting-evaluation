#!/usr/bin/env python3
"""
Evidence for METHOD.md. Runs offline in about ten seconds.

  python scale_invariance.py

Produces the table that goes into the paper's ablation section and the numbers
that answer reviewer question Q2.
"""

import inspect

import numpy as np
import sklearn.discriminant_analysis as da
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED, N, P, B = 0, 400, 40, 5
W = np.array([0.05, 0.08, 0.62, 0.15, 0.10])   # a peaked, plausible weight vector
SLICES = [slice(b * (P // B), (b + 1) * (P // B)) for b in range(B)]


def data(seed=SEED, noise=3.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, P))
    y = (X @ rng.normal(0, 0.3, P) + rng.normal(0, noise, N) > 0).astype(int)
    return X, y


def rescale(X, exponent):
    Z = X.copy()
    for b, s in enumerate(SLICES):
        Z[:, s] *= W[b] ** exponent
    return Z


CLASSIFIERS = {
    "LDA (lsqr, shrinkage=auto)": lambda: LDA(solver="lsqr", shrinkage="auto"),
    "LDA (svd, no shrinkage)": lambda: LDA(solver="svd"),
    "Naive Bayes": lambda: GaussianNB(),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=0),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=0),
    "AdaBoost": lambda: AdaBoostClassifier(random_state=0),
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1.0, max_iter=20000),
    "SVM (RBF)": lambda: SVC(kernel="rbf", gamma="scale"),
    "KNN": lambda: KNeighborsClassifier(),
}


def s1_zscore_cancels(X):
    A = StandardScaler().fit_transform(rescale(X, 0.5))
    Bm = StandardScaler().fit_transform(X)
    rel = np.linalg.norm(A - Bm) / np.linalg.norm(Bm)
    print("\n[1] Per-feature z-score cancels per-feature scaling (Eq. 10 + Alg. 2)")
    print(f"    relative Frobenius difference : {rel:.3e}")
    print(f"    -> the weighting is removed before any classifier sees it")
    return rel


def s2_classifier_invariance(X, y):
    print("\n[2] Without the z-score: which classifiers can even see the weights?")
    print(f"    {'classifier':30s} {'behaviour':36s}")
    print("    " + "-" * 66)
    invariant = []
    for name, mk in CLASSIFIERS.items():
        p1 = mk().fit(X, y).predict(X)
        p2 = mk().fit(rescale(X, 0.5), y).predict(rescale(X, 0.5))
        same = np.array_equal(p1, p2)
        invariant.append(same)
        tag = "INVARIANT - weighting has no effect" if same else "scale-sensitive"
        print(f"    {name:30s} {tag:36s}")
    n_inv = sum(invariant)
    print(f"    -> {n_inv}/{len(CLASSIFIERS)} classifiers are structurally invariant")
    return n_inv


def s3_why_lda(): 
    print("\n[3] Why shrinkage-LDA is invariant: scikit-learn standardises internally")
    for line in inspect.getsource(da._cov).splitlines():
        if any(k in line for k in ("StandardScaler", "fit_transform", "scale_",
                                   "ledoit_wolf")):
            print(f"      {line.strip()}")


def s4_unregularised(X, y):
    print("\n[4] An unpenalised linear model is a reparameterisation, not a change")
    mk = lambda: LogisticRegression(C=1e10, max_iter=500000, tol=1e-12)
    a = mk().fit(X, y).predict_proba(X)[:, 1]
    b = mk().fit(rescale(X, 0.5), y).predict_proba(rescale(X, 0.5))[:, 1]
    d = np.abs(a - b).max()
    print(f"    max |delta p| = {d:.3e}")
    print("    -> the downstream penalty is what makes weighting act (see METHOD 2.1)")
    return d


def s5_group_ridge_equivalence(X, y, lam=2.0, gamma=1.0):
    print("\n[5] Adaptive group ridge (M1) == plain L2 on the rescaled design (M2)")
    pen = np.zeros(P)
    for b, s in enumerate(SLICES):
        pen[s] = W[b] ** (-gamma)
    beta_direct = np.linalg.solve(X.T @ X + lam * np.diag(pen), X.T @ y)

    Xt = rescale(X, gamma / 2)
    alpha = np.linalg.solve(Xt.T @ Xt + lam * np.eye(P), Xt.T @ y)
    beta_repar = alpha.copy()
    for b, s in enumerate(SLICES):
        beta_repar[s] = W[b] ** (gamma / 2) * alpha[s]

    d = np.abs(beta_direct - beta_repar).max()
    print(f"    max |beta_direct - beta_reparameterised| = {d:.3e}")
    print(f"    equivalence holds: {np.allclose(beta_direct, beta_repar)}")

    print("\n    coefficient mass per band (gamma=1.0):")
    print(f"      {'band':6s} {'w_b':>8s} {'||beta_b|| (M1)':>17s} {'||beta_b|| (ridge)':>20s}")
    plain = np.linalg.solve(X.T @ X + lam * np.eye(P), X.T @ y)
    for b, s in enumerate(SLICES):
        print(f"      {b:<6d} {W[b]:8.3f} {np.linalg.norm(beta_direct[s]):17.4f}"
              f" {np.linalg.norm(plain[s]):20.4f}")
    return d


def main():
    X, y = data()
    print("=" * 72)
    print("Evidence for METHOD.md")
    print(f"  n={N} trials, p={P} features, B={B} bands, w={W.tolist()}")
    print("=" * 72)
    rel = s1_zscore_cancels(X)
    n_inv = s2_classifier_invariance(X, y)
    s3_why_lda()
    s4_unregularised(X, y)
    d5 = s5_group_ridge_equivalence(X, y)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  z-score cancels the weighting          : {rel < 1e-10}  ({rel:.1e})")
    print(f"  classifiers blind to it even without   : {n_inv}/{len(CLASSIFIERS)}")
    print(f"  (M1) == (M2) to machine precision      : {d5 < 1e-12}  ({d5:.1e})")
    print("\n  The published mechanism cannot act. The regularisation form can.")
    print("=" * 72)


if __name__ == "__main__":
    main()
