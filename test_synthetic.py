"""
Offline validation. Runs without MOABB and without any data download.

Generates synthetic filter-bank data in which exactly ONE band carries a
class-dependent spatial variance difference, then checks that:

  1. CSP recovers it,
  2. the MI weighting puts its mass on that band,
  3. the published sqrt(w) + z-score pipeline produces the SAME feature matrix
     as no weighting at all  <-- the finding that drives the whole revision,
  4. the adaptive-regularisation variant does not,
  5. the statistics and the Go/No-Go verdict behave correctly.

Run:  python test_synthetic.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from awfbcsp import (
    CSP,
    AWFBCSP,
    add_relative_noise,
    cd_diagram,
    compare_all,
    format_verdict,
    friedman_nemenyi,
    make_pipelines,
    noise_calibration,
    verdict,
    weighting_is_effective,
)
from awfbcsp.features import FilterBankCSP, MIBandWeights

RNG = np.random.default_rng(0)
PASS, FAIL = "  [ok]  ", "  [FAIL]"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {name}{('  ' + detail) if detail else ''}")


def make_data(n_trials=120, n_ch=12, n_times=500, n_bands=5,
              signal_band=2, effect=1.6, snr=0.30, seed=0):
    """Only ``signal_band`` carries class information.

    The discriminative source is blended into a shared background at ``snr`` so
    the problem is not linearly separable to machine precision -- otherwise
    every pipeline saturates at 100% and the comparison says nothing.
    """
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_trials // 2), np.ones(n_trials - n_trials // 2)].astype(int)
    rng.shuffle(y)
    X = rng.normal(0, 1, (n_trials, n_ch, n_times, n_bands))
    mix = rng.normal(0, 1, (n_ch, n_ch))
    for i in range(n_trials):
        src = rng.normal(0, 1, (n_ch, n_times))
        gain = np.ones(n_ch)
        jitter = rng.normal(1.0, 0.25)          # per-trial effect variability
        e = max(effect * jitter, 1.02)
        gain[0] = e if y[i] == 0 else 1.0 / e
        gain[1] = 1.0 / e if y[i] == 0 else e
        signal = mix @ (src * gain[:, None])
        signal /= signal.std() + 1e-12
        background = X[i, :, :, signal_band]
        X[i, :, :, signal_band] = snr * signal + (1 - snr) * background
    return X, y, signal_band


def main():
    print("\n=== synthetic data ===")
    X, y, signal_band = make_data()
    print(f"  X {X.shape}  y balance {np.bincount(y)}  signal in band {signal_band}")

    # -- 1. CSP ------------------------------------------------------------
    print("\n=== 1. CSP ===")
    acc_sig = cross_val_score(
        __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
            ("csp", CSP(n_components=3)),
            ("lda", __import__("sklearn.discriminant_analysis",
                               fromlist=["LinearDiscriminantAnalysis"])
             .LinearDiscriminantAnalysis()),
        ]), X[..., signal_band], y, cv=5).mean()
    acc_noise = cross_val_score(
        __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
            ("csp", CSP(n_components=3)),
            ("lda", __import__("sklearn.discriminant_analysis",
                               fromlist=["LinearDiscriminantAnalysis"])
             .LinearDiscriminantAnalysis()),
        ]), X[..., 0], y, cv=5).mean()
    check("CSP recovers the informative band", 0.62 < acc_sig < 0.99, f"acc={acc_sig:.3f}")
    check("CSP finds nothing in a noise band", acc_noise < 0.65, f"acc={acc_noise:.3f}")

    # -- 2. filter bank ----------------------------------------------------
    print("\n=== 2. filter bank ===")
    fb = FilterBankCSP(n_components=3).fit(X, y)
    F = fb.transform(X)
    check("feature dimension is 2*m*n_bands", F.shape == (X.shape[0], 6 * 5),
          f"{F.shape}")
    check("band slices cover every column",
          fb.band_slices_[-1].stop == F.shape[1])

    # -- 3. MI weighting: descriptor comparison -----------------------------
    print("\n=== 3. MI band weighting: which descriptor finds the right band? ===")
    fb3 = FilterBankCSP(n_components=3).fit(X, y)
    F3 = fb3.transform(X)
    found = {}
    for desc in ("logvar_li", "logpower", "csp"):
        for tau in (0.1, 0.02):
            kw = dict(features=F3, band_slices=fb3.band_slices_) if desc == "csp" else {}
            mw = MIBandWeights(tau=tau, descriptor=desc).fit(X, y, **kw)
            hit = int(np.argmax(mw.weights_)) == signal_band
            found[(desc, tau)] = hit
            print(f"       {desc:10s} tau={tau:<5} "
                  f"{'correct   ' if hit else 'WRONG BAND'}  "
                  f"max_w={mw.weights_.max():.3f}  H={mw.weight_entropy_:.3f}  "
                  f"w={np.round(mw.weights_, 3).tolist()}")
    mw = MIBandWeights(tau=0.1).fit(X, y)
    check("weights sum to 1", np.isclose(mw.weights_.sum(), 1.0))
    check("aggregate descriptors (manuscript style) miss the informative band",
          not found[("logvar_li", 0.1)] and not found[("logvar_li", 0.02)],
          "-> and a low tau concentrates weight on the WRONG band")
    check("CSP-feature descriptor finds it at both temperatures",
          found[("csp", 0.1)] and found[("csp", 0.02)])

    # -- 4. THE DIAGNOSTIC -------------------------------------------------
    print("\n=== 4. does sqrt(w) survive the z-score? (reviewer Q2) ===")
    d = weighting_is_effective(X, y, n_components=3, use_interaction=False)
    print(f"       relative Frobenius difference : {d['relative_frobenius_diff']:.3e}")
    print(f"       max absolute difference       : {d['max_abs_diff']:.3e}")
    check("published sqrt(w) pipeline is numerically inert",
          not d["effective"],
          "-> weighting removed by per-feature standardisation")

    Fp = AWFBCSP(variant="published", n_components=3,
                 use_interaction=False).fit(X, y).transform(X)
    Fr = AWFBCSP(variant="reg", n_components=3,
                 use_interaction=False).fit(X, y).transform(X)
    Fn = AWFBCSP(variant="noweight", n_components=3,
                 use_interaction=False).fit(X, y).transform(X)
    rel_reg = np.linalg.norm(Fr - Fn) / np.linalg.norm(Fn)
    check("adaptive-regularisation variant is NOT inert", rel_reg > 1e-3,
          f"relative difference {rel_reg:.3e}")
    check("published and noweight are identical to float precision",
          np.allclose(Fp, Fn, atol=1e-10))

    # -- 5. pipelines ------------------------------------------------------
    print("\n=== 5. pipelines run end to end ===")
    pipes = make_pipelines(n_components=3, clf="lda")
    accs = {}
    for name, pipe in pipes.items():
        accs[name] = cross_val_score(pipe, X, y, cv=5).mean()
        print(f"       {name:22s} {accs[name]:.4f}")
    check("every pipeline produced a score", len(accs) == len(pipes),
          f"{len(accs)} pipelines")
    check("matched control FBCSP-L2 is present", "FBCSP-L2" in accs,
          "-> the correct reference for AWFBCSP-reg (same classifier, gamma=0)")
    check("published and noweight score identically",
          abs(accs["AWFBCSP-published"] - accs["AWFBCSP-noweight"]) < 1e-9,
          f"delta={abs(accs['AWFBCSP-published']-accs['AWFBCSP-noweight']):.2e}")

    # -- 6. noise calibration ----------------------------------------------
    print("\n=== 6. noise calibration (reviewer Q3) ===")
    volts = X * 3e-5
    for tag, arr in (("volts", volts), ("microvolts", X * 30.0)):
        c = noise_calibration(arr, 0.05)
        print(f"       sigma=0.05 on {tag:11s}: {c['sigma_relative_to_signal_sd']:10.4g}"
              f" x signal sd, SNR {c['snr_db']:+8.1f} dB")
    c_v = noise_calibration(volts, 0.05)
    c_u = noise_calibration(X * 30.0, 0.05)
    check("the same sigma means opposite things in the two unit systems",
          c_v["snr_db"] < -20 < 20 < c_u["snr_db"])
    Xn = add_relative_noise(X, snr_db=-6.0, rng=1)
    check("relative noise injection preserves shape", Xn.shape == X.shape)
    acc_noisy = cross_val_score(pipes["FBCSP"], Xn, y, cv=5).mean()
    check("relative noise degrades accuracy", acc_noisy < accs["FBCSP"],
          f"{accs['FBCSP']:.3f} -> {acc_noisy:.3f}")

    # -- 7. statistics -----------------------------------------------------
    print("\n=== 7. statistics ===")
    rng = np.random.default_rng(7)
    n_sub = 30
    base = rng.normal(0.70, 0.06, n_sub)
    rows = []
    for s in range(n_sub):
        rows += [
            {"subject": s, "pipeline": "FBCSP", "score": base[s]},
            {"subject": s, "pipeline": "AWFBCSP-published",
             "score": base[s] + rng.normal(0.0, 0.004)},
            {"subject": s, "pipeline": "AWFBCSP-reg",
             "score": base[s] + rng.normal(0.022, 0.018)},
            {"subject": s, "pipeline": "CSP",
             "score": base[s] - rng.normal(0.030, 0.020)},
        ]
    scores = pd.DataFrame(rows)

    tbl = compare_all(scores, reference="FBCSP")
    print(tbl[["method", "mean_diff", "p_value", "p_holm",
               "significant", "effect"]].to_string(index=False))
    check("reg is significantly better than FBCSP",
          bool(tbl.set_index("method").loc["AWFBCSP-reg", "significant"]))
    row = tbl.set_index("method").loc["AWFBCSP-published"]
    check("published is NOT significantly better than FBCSP",
          not (bool(row["significant"]) and row["mean_diff"] > 0),
          f"mean_diff={row['mean_diff']*100:+.3f} pts, p_holm={row['p_holm']:.3f}")

    nem = friedman_nemenyi(scores)
    check("Friedman test runs", nem is not None and nem["significant"],
          f"CD={nem['critical_difference']:.3f}")
    path = cd_diagram(nem, "cd_diagram_synthetic.png", title="synthetic check")
    check("CD diagram written", path is not None)

    # -- 8. verdict --------------------------------------------------------
    print("\n=== 8. Go/No-Go verdict logic ===")
    v_bad = verdict(scores, "AWFBCSP-published", "FBCSP", diag=d)
    print(format_verdict(v_bad))
    check("inert method does not get a GO", v_bad["verdict"] != "GO",
          v_bad["verdict"])
    v_good = verdict(scores, "AWFBCSP-reg", "FBCSP",
                     diag={"effective": True, "relative_frobenius_diff": 1.0})
    check("working method gets a GO", v_good["verdict"] == "GO", v_good["verdict"])

    # -- summary -----------------------------------------------------------
    n_pass, n_tot = sum(_results), len(_results)
    print(f"\n{'=' * 70}\n{n_pass}/{n_tot} checks passed\n{'=' * 70}")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
