"""
Diagnostics that answer reviewer questions directly with numbers.

D1  weighting_is_effective  -> reviewer question Q2 (does sqrt(w) survive z-score?)
D2  noise_calibration        -> reviewer question Q3 (what is sigma in signal units?)
D3  weight_summary           -> experiment E10 (weight peakedness vs gain)
"""

import numpy as np

from .pipelines import AWFBCSP


def weighting_is_effective(X, y, n_components=4, tau=0.1, random_state=0,
                           use_interaction=False):
    """D1. Compare the feature matrices of 'published' and 'noweight'.

    'published' applies sqrt(w) per band and then z-scores each feature.
    'noweight' only z-scores. Because per-feature standardisation removes any
    per-feature constant multiplier, these two matrices are expected to be
    numerically identical up to floating point -- which would mean the weighting
    mechanism has no effect on any scale-sensitive classifier.

    Set use_interaction=False to isolate the weighting: the 4-D interaction
    block is present in both variants and is not affected by the weights, so
    leaving it in only dilutes the measurement.

    Returns a dict; ``effective`` is the verdict.
    """
    kw = dict(n_components=n_components, tau=tau, random_state=random_state,
              use_interaction=use_interaction)
    Fw = AWFBCSP(variant="published", **kw).fit(X, y).transform(X)
    Fn = AWFBCSP(variant="noweight", **kw).fit(X, y).transform(X)

    if Fw.shape != Fn.shape:
        raise RuntimeError("variants produced different feature dimensions")

    diff = np.abs(Fw - Fn)
    scale = np.abs(Fn).mean() + 1e-12
    rel_fro = float(np.linalg.norm(Fw - Fn) / (np.linalg.norm(Fn) + 1e-12))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_frobenius_diff": rel_fro,
        "feature_scale": float(scale),
        # 1e-8 is far above float64 round-off but far below any real effect
        "effective": bool(rel_fro > 1e-8),
    }


def noise_calibration(X, sigma_absolute):
    """D2. Express an absolute noise sigma in signal-relative terms.

    The manuscript injects N(0, sigma^2) with sigma in 0.05..0.30 without
    stating the units of X. This reports the implied SNR so the experiment can
    be re-specified in dB (see E6).

    X may be 3-D or 4-D; the band axis is collapsed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 4:
        X = X[..., 0]
    sd = float(X.std())
    snr = sd / max(sigma_absolute, 1e-30)
    return {
        "signal_sd": sd,
        "sigma_absolute": float(sigma_absolute),
        "sigma_relative_to_signal_sd": float(sigma_absolute / (sd + 1e-30)),
        "snr_db": float(20 * np.log10(max(snr, 1e-30))),
        "likely_units": "volts" if sd < 1e-3 else "microvolts_or_arbitrary",
    }


def add_relative_noise(X, snr_db, rng=None):
    """Inject noise at a stated SNR (dB) instead of an absolute sigma.

    Noise sd is computed per trial and per channel from that channel's own sd,
    so the perturbation means the same thing across datasets with different
    amplitude scaling. This is the fix for issue P0-2.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    axis = 2  # time axis, for both 3-D and 4-D layouts
    sd = X.std(axis=axis, keepdims=True)
    noise_sd = sd / (10 ** (snr_db / 20.0))
    return X + rng.normal(0.0, 1.0, size=X.shape) * noise_sd


def weight_summary(fitted_pipeline):
    """D3. Pull band weights and entropy out of a fitted pipeline."""
    feat = fitted_pipeline.named_steps.get("feat")
    return feat.report() if feat is not None else {}
