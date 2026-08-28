"""
Pipelines under test.

The four AWFBCSP variants exist to answer one question that the manuscript
cannot currently answer (reviewer question Q2): does the sqrt(w) band weighting
survive the z-score standardisation that Algorithm 2 applies immediately after?

  published : FBCSP -> sqrt(w) scaling -> [+interaction] -> z-score
  noweight  : FBCSP ->                    [+interaction] -> z-score
  reg       : FBCSP -> z-score -> w^(gamma/2) scaling      (adaptive group ridge)
  fbcsp     : FBCSP -> z-score                             (plain baseline)

'published' and 'noweight' differ only by a per-feature constant factor applied
BEFORE per-feature standardisation. If the mechanism works as described, their
outputs must differ. diagnostics.weighting_is_effective() measures exactly that.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import (
    CSP,
    CrossBandInteraction,
    FilterBankCSP,
    MIBandWeights,
    MIBIFSelect,
)

VARIANTS = ("published", "noweight", "reg", "fbcsp")


class AWFBCSP(BaseEstimator, TransformerMixin):
    """Adaptive weighted filter-bank CSP. Input: (n_trials, n_ch, n_times, n_bands)."""

    def __init__(self, variant="published", n_components=4, tau=0.015, gamma=1.0,
                 shrinkage="ledoit_wolf", descriptor="logvar_li",
                 use_interaction=True, n_neighbors=3, random_state=0):
        self.variant = variant
        self.n_components = n_components
        self.tau = tau
        self.gamma = gamma
        self.shrinkage = shrinkage
        self.descriptor = descriptor
        self.use_interaction = use_interaction
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    # -- internals ---------------------------------------------------------

    def _weight_exponent(self):
        return 0.5 if self.variant == "published" else self.gamma / 2.0

    def fit(self, X, y):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.fb_ = FilterBankCSP(self.n_components, self.shrinkage).fit(X, y)
        F = self.fb_.transform(X)

        self.weighter_ = None
        self.inter_ = None
        if self.variant in ("published", "reg"):
            self.weighter_ = MIBandWeights(
                tau=self.tau, descriptor=self.descriptor,
                n_neighbors=self.n_neighbors, random_state=self.random_state,
            ).fit(X, y, features=F, band_slices=self.fb_.band_slices_)
        if self.use_interaction and self.variant in ("published", "noweight"):
            ref = self.weighter_ or MIBandWeights(
                tau=self.tau, descriptor=self.descriptor,
                n_neighbors=self.n_neighbors, random_state=self.random_state,
            ).fit(X, y, features=F, band_slices=self.fb_.band_slices_)
            self.inter_ = CrossBandInteraction(ref).fit(X)

        if self.variant == "reg":
            # standardise FIRST, then weight -- the weighting survives.
            self.scaler_ = StandardScaler().fit(F)
        else:
            # reproduce Algorithm 2: weight (if any), append interaction, then
            # standardise. The scaler is fit on the already-weighted matrix.
            self.scaler_ = StandardScaler().fit(self._pre_scale(F, X))
        return self

    def _pre_scale(self, F, X):
        """Everything Algorithm 2 does before its final z-score."""
        out = F
        if self.variant == "published":
            out = self._apply_weights(out)
        if self.inter_ is not None:
            out = np.hstack([out, self.inter_.transform(X)])
        return out

    def _apply_weights(self, F):
        F = F.copy()
        e = self._weight_exponent()
        for b, sl in enumerate(self.fb_.band_slices_):
            F[:, sl] *= self.weighter_.weights_[b] ** e
        return F

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        F = self.fb_.transform(X)
        if self.variant == "reg":
            return self._apply_weights(self.scaler_.transform(F))
        return self.scaler_.transform(self._pre_scale(F, X))

    # -- reporting ---------------------------------------------------------

    def report(self):
        """Per-fit values worth logging: band weights, MI, entropy."""
        if self.weighter_ is None:
            return {}
        w = self.weighter_
        return {
            "weights": w.weights_.tolist(),
            "mi": w.mi_.tolist(),
            "weight_entropy": w.weight_entropy_,
            "uniform_entropy": w.uniform_entropy_,
            "max_weight": float(w.weights_.max()),
        }


# --------------------------------------------------------------------------
# classifiers
# --------------------------------------------------------------------------

def _clf(name, random_state=0):
    if name == "lda":
        return LDA(solver="lsqr", shrinkage="auto")
    if name == "lr":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    raise ValueError(f"unknown classifier: {name}")


def make_pipelines(n_components=4, tau=0.015, gamma=1.0, clf="lda",
                   mibif_k=4, random_state=0):
    """Filter-bank pipelines. Use with FilterBankLeftRightImagery."""
    base = dict(n_components=n_components, tau=tau, gamma=gamma,
                random_state=random_state)
    p = {}
    p["FBCSP"] = Pipeline([
        ("feat", AWFBCSP(variant="fbcsp", use_interaction=False, **base)),
        ("clf", _clf(clf, random_state)),
    ])
    p["FBCSP+MIBIF"] = Pipeline([
        ("feat", AWFBCSP(variant="fbcsp", use_interaction=False, **base)),
        ("sel", MIBIFSelect(k=mibif_k, random_state=random_state)),
        ("clf", _clf(clf, random_state)),
    ])
    p["AWFBCSP-published"] = Pipeline([
        ("feat", AWFBCSP(variant="published", use_interaction=True, **base)),
        ("clf", _clf(clf, random_state)),
    ])
    p["AWFBCSP-noweight"] = Pipeline([
        ("feat", AWFBCSP(variant="noweight", use_interaction=True, **base)),
        ("clf", _clf(clf, random_state)),
    ])
    # matched control for AWFBCSP-reg: identical pipeline with gamma=0, i.e.
    # uniform weights. Comparing AWFBCSP-reg against FBCSP would confound the
    # weighting with the change of classifier; this isolates the weighting.
    p["FBCSP-L2"] = Pipeline([
        ("feat", AWFBCSP(variant="reg", use_interaction=False,
                         **{**base, "gamma": 0.0})),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
    ])
    p["AWFBCSP-reg"] = Pipeline([
        ("feat", AWFBCSP(variant="reg", use_interaction=False, **base)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
    ])
    return p


def make_broadband_pipelines(n_components=4, clf="lda", random_state=0):
    """Broadband pipelines. Use with LeftRightImagery (3-D input)."""
    p = {"CSP": Pipeline([
        ("csp", CSP(n_components=n_components)),
        ("sc", StandardScaler()),
        ("clf", _clf(clf, random_state)),
    ])}
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        p["TS+LR"] = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ])
    except ImportError:
        pass
    return p
