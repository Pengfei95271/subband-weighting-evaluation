"""
Experiments E2, E3, E6, E10 and the sensitivity sweeps.

Every function takes ``subjects``: a dict mapping subject id -> (X, y), where X
is filter-bank data of shape (n_trials, n_channels, n_times, n_bands). Build it
once with moabb and reuse it; see run_experiments.py.

All returns are tidy DataFrames ready for awfbcsp.stats.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .diagnostics import add_relative_noise
from .features import FilterBankCSP, MIBandWeights
from .pipelines import make_pipelines


def _cv(pipe, X, y, n_splits=5, seed=0):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, X, y, cv=cv, n_jobs=1).mean())


def _subsample(X, y, n, rng):
    """Class-balanced subsample of n trials."""
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        take = min(n // len(np.unique(y)), len(pool))
        idx.append(rng.choice(pool, take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------
# E2 -- learning curve
# --------------------------------------------------------------------------

def e2_learning_curve(subjects, fractions=(0.1, 0.2, 0.4, 0.7, 1.0),
                      pipelines=None, seeds=(0, 1, 2), **pipe_kw):
    """Accuracy versus training-set size.

    This is the experiment the whole positioning depends on: an
    information-theoretic prior should beat a purely learned method when data is
    scarce. If no crossover exists, the data-efficiency claim dies.
    """
    pipes = pipelines or make_pipelines(**pipe_kw)
    rows = []
    for sid, (X, y) in subjects.items():
        n_total = len(y)
        for frac in fractions:
            n = max(int(round(frac * n_total)), 20)
            if n > n_total:
                continue
            for seed in seeds:
                rng = np.random.default_rng(seed)
                Xs, ys = _subsample(X, y, n, rng)
                if len(np.unique(ys)) < 2 or np.bincount(ys).min() < 3:
                    continue
                for name, pipe in pipes.items():
                    rows.append(dict(subject=sid, fraction=frac, n_trials=len(ys),
                                     seed=seed, pipeline=name,
                                     score=_cv(pipe, Xs, ys, seed=seed)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E3 / E3b -- ablation
# --------------------------------------------------------------------------

def e3_ablation(subjects, seeds=(0, 1, 2), **pipe_kw):
    """All variants side by side, including the published/noweight control pair.

    The two must score identically if the published mechanism is inert. Report
    this row explicitly rather than hiding it.
    """
    pipes = make_pipelines(**pipe_kw)
    rows = []
    for sid, (X, y) in subjects.items():
        for seed in seeds:
            for name, pipe in pipes.items():
                rows.append(dict(subject=sid, seed=seed, pipeline=name,
                                 score=_cv(pipe, X, y, seed=seed)))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["subject", "seed"], columns="pipeline",
                         values="score")
    if {"AWFBCSP-published", "AWFBCSP-noweight"} <= set(piv.columns):
        d = (piv["AWFBCSP-published"] - piv["AWFBCSP-noweight"]).abs()
        df.attrs["published_minus_noweight_max_abs"] = float(d.max())
        df.attrs["mechanism_effective"] = bool(d.max() > 1e-9)
    return df


# --------------------------------------------------------------------------
# sensitivity sweeps
# --------------------------------------------------------------------------

def sweep_tau(subjects, taus=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0),
              descriptor="csp", n_components=4, seed=0):
    """tau sensitivity. tau -> infinity makes the weights uniform, i.e. FBCSP.

    Also records the weight entropy, so the sweep doubles as a check that the
    reported example weight vector is reachable at a sensible temperature.
    """
    rows = []
    for sid, (X, y) in subjects.items():
        fb = FilterBankCSP(n_components=n_components).fit(X, y)
        F = fb.transform(X)
        for tau in taus:
            kw = dict(features=F, band_slices=fb.band_slices_) \
                if descriptor == "csp" else {}
            mw = MIBandWeights(tau=tau, descriptor=descriptor,
                               random_state=seed).fit(X, y, **kw)
            rows.append(dict(subject=sid, tau=tau, descriptor=descriptor,
                             entropy=mw.weight_entropy_,
                             uniform_entropy=mw.uniform_entropy_,
                             max_weight=float(mw.weights_.max()),
                             argmax_band=int(np.argmax(mw.weights_)),
                             weights=mw.weights_.tolist()))
    return pd.DataFrame(rows)


def sweep_gamma(subjects, gammas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                seeds=(0, 1, 2), **pipe_kw):
    """gamma sensitivity for the adaptive-regularisation variant.

    gamma = 0 is plain ridge on filter-bank features, i.e. FBCSP. A non-trivial
    optimum away from 0 is the evidence that the MI prior carries information.
    """
    rows = []
    for sid, (X, y) in subjects.items():
        for g in gammas:
            pipe = make_pipelines(gamma=g, **pipe_kw)["AWFBCSP-reg"]
            for seed in seeds:
                rows.append(dict(subject=sid, gamma=g, seed=seed,
                                 score=_cv(pipe, X, y, seed=seed)))
    return pd.DataFrame(rows)


def compare_descriptors(subjects, descriptors=("logvar_li", "logpower", "csp"),
                        taus=(0.1, 0.02), n_components=4, seed=0):
    """Which descriptor produces which weights. Feeds METHOD.md section 3."""
    rows = []
    for sid, (X, y) in subjects.items():
        fb = FilterBankCSP(n_components=n_components).fit(X, y)
        F = fb.transform(X)
        for desc in descriptors:
            for tau in taus:
                kw = dict(features=F, band_slices=fb.band_slices_) \
                    if desc == "csp" else {}
                mw = MIBandWeights(tau=tau, descriptor=desc,
                                   random_state=seed).fit(X, y, **kw)
                rows.append(dict(subject=sid, descriptor=desc, tau=tau,
                                 argmax_band=int(np.argmax(mw.weights_)),
                                 max_weight=float(mw.weights_.max()),
                                 entropy=mw.weight_entropy_,
                                 weights=mw.weights_.tolist()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E6 -- noise robustness, defined in relative SNR
# --------------------------------------------------------------------------

def e6_noise(subjects, snr_db=(30, 20, 12, 6, 0, -6), seeds=(0, 1, 2),
             pipelines=None, **pipe_kw):
    """Degradation versus SNR in dB.

    Replaces the manuscript's absolute sigma, which is unitless and therefore
    means different things on datasets with different amplitude scaling.
    """
    pipes = pipelines or make_pipelines(**pipe_kw)
    rows = []
    for sid, (X, y) in subjects.items():
        for snr in snr_db:
            for seed in seeds:
                Xn = add_relative_noise(X, snr_db=snr, rng=seed)
                for name, pipe in pipes.items():
                    rows.append(dict(subject=sid, snr_db=snr, seed=seed,
                                     pipeline=name,
                                     score=_cv(pipe, Xn, y, seed=seed)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E10 -- does the gain track weight peakedness?
# --------------------------------------------------------------------------

def e10_entropy_vs_gain(subjects, method="AWFBCSP-reg", reference="FBCSP-L2",
                        seeds=(0, 1, 2), descriptor="csp", tau=0.1,
                        n_components=4, **pipe_kw):
    """Hypothesis H1: the gain grows as the weight distribution leaves uniform.

    If this holds, near-uniform weights and near-zero average gain stop being an
    embarrassment and become the predicted behaviour of a method that only acts
    when there is band specificity to act on.
    """
    pipes = make_pipelines(tau=tau, n_components=n_components, **pipe_kw)
    rows = []
    for sid, (X, y) in subjects.items():
        fb = FilterBankCSP(n_components=n_components).fit(X, y)
        F = fb.transform(X)
        kw = dict(features=F, band_slices=fb.band_slices_) \
            if descriptor == "csp" else {}
        mw = MIBandWeights(tau=tau, descriptor=descriptor).fit(X, y, **kw)
        a = np.mean([_cv(pipes[method], X, y, seed=s) for s in seeds])
        b = np.mean([_cv(pipes[reference], X, y, seed=s) for s in seeds])
        rows.append(dict(subject=sid, entropy=mw.weight_entropy_,
                         uniform_entropy=mw.uniform_entropy_,
                         entropy_deficit=mw.uniform_entropy_ - mw.weight_entropy_,
                         max_weight=float(mw.weights_.max()),
                         score_method=a, score_reference=b, gain=a - b))
    df = pd.DataFrame(rows)
    if len(df) >= 4:
        from scipy.stats import spearmanr
        r_h, p_h = spearmanr(df["entropy"], df["gain"])
        r_d, p_d = spearmanr(df["entropy_deficit"], df["gain"])
        df.attrs["spearman_entropy_vs_gain"] = (float(r_h), float(p_h))
        df.attrs["spearman_deficit_vs_gain"] = (float(r_d), float(p_d))
        df.attrs["H1_supported"] = bool(r_d > 0 and p_d < 0.05)
    return df


def summarise_e10(df):
    lines = ["H1: gain increases as weights depart from uniform", "-" * 55]
    r, p = df.attrs.get("spearman_deficit_vs_gain", (float("nan"),) * 2)
    rh, ph = df.attrs.get("spearman_entropy_vs_gain", (float("nan"),) * 2)
    lines += [f"  n subjects                       : {len(df)}",
              f"  Spearman(entropy deficit, gain)  : rho={r:+.3f}  p={p:.4f}",
              f"  Spearman(entropy, gain)          : rho={rh:+.3f}  p={ph:.4f}",
              f"  mean gain                        : {df['gain'].mean()*100:+.2f} pts",
              f"  H1 supported                     : {df.attrs.get('H1_supported')}"]
    return "\n".join(lines)
