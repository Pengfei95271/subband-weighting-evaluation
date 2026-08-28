"""
Spatial-spectral feature extraction for the Go/No-Go study.

Deliberately self-contained: depends only on numpy / scipy / sklearn so that every
transformer here can be unit-tested offline, without MOABB or a data download.

Input convention
----------------
Broadband estimators take X of shape (n_trials, n_channels, n_times).
Filter-bank estimators take X of shape (n_trials, n_channels, n_times, n_bands),
which is exactly what moabb.paradigms.FilterBankLeftRightImagery returns.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas
from sklearn.feature_selection import mutual_info_classif


# --------------------------------------------------------------------------
# covariance
# --------------------------------------------------------------------------

def _cov(X, shrinkage="ledoit_wolf"):
    """Trace-normalised covariance of one trial. X: (n_channels, n_times)."""
    C = X @ X.T
    tr = np.trace(C)
    if tr <= 0 or not np.isfinite(tr):
        return np.eye(X.shape[0])
    C = C / tr
    if shrinkage == "none":
        return C
    Z = X.T  # (n_times, n_channels), samples-by-features for sklearn
    if shrinkage == "ledoit_wolf":
        Cs, _ = ledoit_wolf(Z, assume_centered=True)
    elif shrinkage == "oas":
        Cs, _ = oas(Z, assume_centered=True)
    else:
        raise ValueError(f"unknown shrinkage: {shrinkage}")
    tr = np.trace(Cs)
    return Cs / tr if tr > 0 else C


def _class_cov(X, y, label, shrinkage):
    idx = np.flatnonzero(y == label)
    if idx.size == 0:
        return np.eye(X.shape[1])
    return np.mean([_cov(X[i], shrinkage) for i in idx], axis=0)


# --------------------------------------------------------------------------
# CSP
# --------------------------------------------------------------------------

class CSP(BaseEstimator, TransformerMixin):
    """Binary Common Spatial Pattern with log-variance features.

    Parameters
    ----------
    n_components : int
        Number of filter PAIRS (m in the manuscript). Output dimension is 2*m.
    shrinkage : {'ledoit_wolf', 'oas', 'none'}
        Covariance regularisation. The manuscript says "optionally with
        shrinkage" without specifying; this makes the choice explicit and
        auditable (issue P1-1).
    """

    def __init__(self, n_components=4, shrinkage="ledoit_wolf"):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("CSP here is binary only; use OVR for multiclass.")
        n_ch = X.shape[1]
        m = min(self.n_components, n_ch // 2)
        if m < 1:
            raise ValueError("n_components too small for this montage")
        self.m_ = m

        C0 = _class_cov(X, y, self.classes_[0], self.shrinkage)
        C1 = _class_cov(X, y, self.classes_[1], self.shrinkage)
        Csum = C0 + C1
        Csum += 1e-10 * np.trace(Csum) / n_ch * np.eye(n_ch)

        evals, evecs = eigh(C0, Csum)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        self.filters_ = np.concatenate([evecs[:, :m], evecs[:, -m:]], axis=1)  # (n_ch, 2m)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.einsum("cf,nct->nft", self.filters_, X)  # (n_trials, 2m, n_times)
        var = Z.var(axis=2)
        var = np.maximum(var, 1e-20)
        var = var / var.sum(axis=1, keepdims=True)
        return np.log(var)


# --------------------------------------------------------------------------
# filter-bank CSP features
# --------------------------------------------------------------------------

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Per-band CSP, concatenated. X: (n_trials, n_channels, n_times, n_bands).

    Exposes ``band_slices_`` so downstream steps can address each band's block
    of columns -- needed for band weighting and for group-wise regularisation.
    """

    def __init__(self, n_components=4, shrinkage="ledoit_wolf"):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError(f"expected 4-D filter-bank input, got shape {X.shape}")
        self.n_bands_ = X.shape[3]
        self.csps_ = []
        for b in range(self.n_bands_):
            csp = CSP(self.n_components, self.shrinkage).fit(X[..., b], y)
            self.csps_.append(csp)
        w = 2 * self.csps_[0].m_
        self.band_slices_ = [slice(b * w, (b + 1) * w) for b in range(self.n_bands_)]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.concatenate(
            [self.csps_[b].transform(X[..., b]) for b in range(self.n_bands_)], axis=1
        )


# --------------------------------------------------------------------------
# mutual-information band weighting
# --------------------------------------------------------------------------

def band_descriptors(X_band, kind="logpower"):
    """Low-dimensional descriptors p^(b) for one band.

    X_band: (n_trials, n_channels, n_times) -> (n_trials, n_descriptors)

    'logpower'   : per-channel log band power
    'logvar_li'  : mean log power, log variance, and a hemispheric laterality
                   index built from the left/right halves of the montage
    """
    power = np.maximum(X_band.var(axis=2), 1e-20)  # (n_trials, n_channels)
    logp = np.log(power)
    if kind == "logpower":
        return logp
    if kind == "logvar_li":
        n_ch = logp.shape[1]
        half = max(n_ch // 2, 1)
        left = logp[:, :half].mean(axis=1)
        right = logp[:, -half:].mean(axis=1)
        li = (left - right) / (np.abs(left) + np.abs(right) + 1e-12)
        return np.column_stack([logp.mean(axis=1), np.log(power.sum(axis=1)), li])
    raise ValueError(f"unknown descriptor kind: {kind}")


class MIBandWeights(BaseEstimator, TransformerMixin):
    """Mutual-information band weights, w = softmax(I_b / tau).

    Every parameter the manuscript leaves unspecified is explicit here
    (issue P1-1): tau, the descriptor set, the MI estimator's k, and how a
    multi-dimensional descriptor is aggregated into one scalar per band.

    Set ``descriptor='csp'`` to score each band on its CSP log-variance
    features instead of aggregate band descriptors. Aggregate descriptors
    (mean power, total power, a coarse laterality index) are largely blind to a
    class difference that lives in the spatial covariance -- which is exactly
    what CSP extracts -- so they can rank the wrong band confidently, and a low
    tau then amplifies that error into a peaked weight vector.

    ``fit`` must only ever see training-fold data -- the pipeline guarantees this.
    """

    def __init__(self, tau=0.015, descriptor="logvar_li", agg="mean",
                 n_neighbors=3, random_state=0):
        self.tau = tau
        self.descriptor = descriptor
        self.agg = agg
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y, features=None, band_slices=None):
        """``features``/``band_slices`` are required when descriptor='csp'."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_bands = X.shape[3]
        mi = np.zeros(n_bands)
        if self.descriptor == "csp":
            if features is None or band_slices is None:
                raise ValueError("descriptor='csp' needs features and band_slices")
            for b, sl in enumerate(band_slices):
                scores = mutual_info_classif(
                    features[:, sl], y, n_neighbors=self.n_neighbors,
                    random_state=self.random_state)
                mi[b] = scores.mean() if self.agg == "mean" else scores.max()
        else:
            for b in range(n_bands):
                p = band_descriptors(X[..., b], self.descriptor)
                scores = mutual_info_classif(
                    p, y, n_neighbors=self.n_neighbors,
                    random_state=self.random_state)
                mi[b] = scores.mean() if self.agg == "mean" else scores.max()
        self.mi_ = mi
        z = mi / max(self.tau, 1e-12)
        z -= z.max()
        w = np.exp(z)
        self.weights_ = w / w.sum()
        # entropy of the weight distribution, in nats; log(n_bands) == uniform.
        # This is the E10 variable: does the gain track weight peakedness?
        p_ = np.maximum(self.weights_, 1e-12)
        self.weight_entropy_ = float(-(p_ * np.log(p_)).sum())
        self.uniform_entropy_ = float(np.log(n_bands))
        return self

    def transform(self, X):
        return X


class ApplyBandWeights(BaseEstimator, TransformerMixin):
    """Scale each band's feature block by weights_[b] ** exponent.

    exponent = 0.5 reproduces the manuscript's sqrt(w) rule (Eq. 10).
    exponent = gamma/2 implements the adaptive group-ridge reparameterisation:
    minimising ||y - X b||^2 + lam * sum_b w_b^-gamma ||b_b||^2 is equivalent to
    plain L2 on the rescaled design X_b <- w_b^(gamma/2) X_b, so a plain
    L2-penalised classifier downstream gives the adaptive-ridge solution.
    """

    def __init__(self, weighter, band_slices_attr="band_slices_", exponent=0.5):
        self.weighter = weighter
        self.band_slices_attr = band_slices_attr
        self.exponent = exponent

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        w = self.weighter.weights_
        slices = self._slices(X.shape[1], len(w))
        for b, sl in enumerate(slices):
            X[:, sl] *= w[b] ** self.exponent
        return X

    @staticmethod
    def _slices(n_features, n_bands):
        width = n_features // n_bands
        return [slice(b * width, (b + 1) * width) for b in range(n_bands)]


class CrossBandInteraction(BaseEstimator, TransformerMixin):
    """The 4-D interaction feature of Eq. (12), appended to the feature vector.

    P1 and P2 are the mean band powers of the two highest-weighted bands. The
    top-2 selection is done from training-fold weights only -- selecting them on
    the full dataset would leak (issue P1-1).
    """

    def __init__(self, weighter, eps=1e-8):
        self.weighter = weighter
        self.eps = eps

    def fit(self, X, y=None):
        self.top2_ = np.argsort(self.weighter.weights_)[::-1][:2]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        b1, b2 = self.top2_
        p1 = np.maximum(X[..., b1].var(axis=2).mean(axis=1), self.eps)
        p2 = np.maximum(X[..., b2].var(axis=2).mean(axis=1), self.eps)
        return np.column_stack([p1 * p2, p1 + p2, np.abs(p1 - p2), p1 / (p2 + self.eps)])


class MIBIFSelect(BaseEstimator, TransformerMixin):
    """Mutual-information best-individual-feature selection.

    This is step 3 of the original FBCSP (Ang et al. 2008/2012). It is the
    baseline the manuscript must beat to claim novelty (issue P0-5): hard top-k
    selection versus the proposed soft weighting.
    """

    def __init__(self, k=4, n_neighbors=3, random_state=0):
        self.k = k
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        mi = mutual_info_classif(
            X, y, n_neighbors=self.n_neighbors, random_state=self.random_state
        )
        k = min(self.k, X.shape[1])
        self.support_ = np.argsort(mi)[::-1][:k]
        self.mi_ = mi
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]
