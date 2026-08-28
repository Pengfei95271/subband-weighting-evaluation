# Method note — why band weighting must enter as regularisation

Paper-ready derivation for the rebuilt method. Every claim here is verified
numerically by `scale_invariance.py`; run it before citing any of it.

Notation: the filter-bank CSP feature vector for trial *i* is
F_i = [F_i^(1)ᵀ, …, F_i^(B)ᵀ]ᵀ ∈ ℝ^{2mB}, with F_i^(b) the log-variance block of
band *b*. Band weights w = (w_1,…,w_B), w_b > 0, Σ w_b = 1. D(w, e) denotes the
diagonal matrix that multiplies every column of block *b* by w_b^e.

---

## 1. The published rule is inert, for two independent reasons

Equation (10) of the manuscript sets F̃^(b) = √w_b · F^(b); Algorithm 2 then
z-scores the concatenated vector.

### 1.1 Per-feature standardisation cancels per-feature scaling

For column *j* with scale factor c_j > 0, the standardised value is

    (c_j F_ij − mean_i(c_j F_ij)) / sd_i(c_j F_ij)
  = c_j (F_ij − mean_i F_ij) / (c_j · sd_i F_ij)
  = (F_ij − mean_i F_ij) / sd_i F_ij,

independent of c_j. So `z(F · D(w, e)) = z(F)` exactly, for any weights and any
exponent. Measured relative Frobenius difference on real-shaped data:
**1.3 × 10⁻¹⁴**, i.e. floating-point round-off.

The weighting is removed before any classifier sees it. This holds for every
classifier, every dataset, every subject.

### 1.2 Most of the classifier suite is invariant even without the z-score

Suppose the standardisation were removed. Scaling is still invisible to:

| classifier | why |
|---|---|
| LDA, shrinkage='auto' | scikit-learn's `_cov` standardises internally before Ledoit–Wolf and rescales afterwards, so the shrunk covariance transforms as DΣD and the projection wᵀx is preserved |
| LDA, no shrinkage | Σ⁻¹(μ₁−μ₀) → D⁻¹Σ⁻¹(μ₁−μ₀); the discriminant wᵀx is invariant |
| Gaussian Naive Bayes | per-feature μ→cμ and σ→cσ; the −log c terms are identical in both classes and cancel in the log-likelihood ratio |
| Decision Tree, Random Forest, Gradient Boosting, AdaBoost | splits depend only on the ordering of feature values, which a positive scaling preserves |

Only logistic regression, SVM, and k-NN are scale-sensitive — and §1.1 disposes
of those too.

**Six of the manuscript's nine classifiers are structurally invariant.** This is
not an implementation detail; it is a property of the estimators. The manuscript's
own Table 2 already shows the fingerprint: on IV-2A, AdaBoost reports
88.92 ± 3.97 for both FBCSP and AWFBCSP, and Decision Tree reports a mean of
81.26 for both. Identical numbers are the expected consequence, not a typo.

### 1.3 What this implies for the reported results

Any non-zero difference between FBCSP and AWFBCSP in the manuscript can only
come from (a) the 4-dimensional cross-band interaction block of Eq. (12), which
is not affected by the weights, or (b) seed variation. It cannot come from the
weighting. This is consistent with the subject-level averages: +0.03 points on
IV-2A and −0.08 points on IV-2B.

---

## 2. The correct formulation: adaptive group regularisation

Band relevance should modulate **how strongly each band's coefficients are
penalised**, not how large its features are. Let β_b be the coefficient block
for band *b*:

> **(M1)**  min_β  L(y, Fβ) + λ Σ_{b=1}^{B} w_b^{−γ} ‖β_b‖₂²

Bands with small w_b get a large penalty w_b^{−γ} and are shrunk towards zero;
bands with large w_b are penalised lightly. γ > 0 replaces the temperature τ as
the sharpness control, and γ = 0 recovers plain ridge — i.e. FBCSP.

### 2.1 Equivalence to a rescaled design

Substitute β_b = w_b^{γ/2} α_b. Then Fβ = Σ_b F^(b) w_b^{γ/2} α_b = F̃α with
F̃^(b) = w_b^{γ/2} F^(b), and w_b^{−γ}‖β_b‖² = w_b^{−γ} w_b^{γ}‖α_b‖² = ‖α_b‖².
So (M1) becomes

> **(M2)**  min_α  L(y, F̃α) + λ‖α‖₂²,   F̃ = F · D(w, γ/2),   β_b = w_b^{γ/2} α_b

**Implementation:** standardise F on the training fold, rescale block *b* by
w_b^{γ/2}, then fit any plain L2-penalised estimator. Verified equal to the
direct solution of (M1) to 1.4 × 10⁻¹⁶.

Two conditions are load-bearing and are exactly what the manuscript violates:

1. **the order** — standardise *then* weight, never the reverse;
2. **the downstream penalty** — the estimator must carry an L2 penalty on the
   coefficients. With an unpenalised estimator, (M2) is a reparameterisation and
   the solution is unchanged. This is why the rebuilt pipeline uses penalised
   logistic regression and not shrinkage-LDA.

### 2.2 Relation to existing filter-bank selection methods

| method | prior on band relevance | selection |
|---|---|---|
| FBCSP + MIBIF (Ang et al.) | mutual information | hard top-k, discrete |
| SFBCSP | sparsity (ℓ₁) | data-driven, discrete |
| SBLFB | sparse Bayesian (ARD) | data-driven, continuous |
| **(M1)** | **mutual information, as a penalty prior** | **continuous, structure-preserving** |

(M1) is an adaptive group ridge in the sense of Zou (2006): the penalty weights
are data-derived rather than fixed. The distinction from SFBCSP/SBLFB is that
the prior is *informational* (MI between band descriptors and labels) rather
than *sparsity-inducing*, so the full filter-bank structure is retained and no
band is deleted. The distinction from MIBIF is soft versus hard selection, which
E4 measures directly by sweeping k.

---

## 3. The descriptor determines whether the weights are even correct

I_b = I(p^(b); y) depends entirely on what p^(b) is. On synthetic data with the
discriminative source placed in band 2 of 5:

| descriptor p^(b) | τ = 0.1 | τ = 0.02 | max weight at τ = 0.02 |
|---|---|---|---|
| mean log power, total power, laterality index (manuscript style) | **wrong band** | **wrong band** | 0.575, on the wrong band |
| per-channel log power | correct | correct | 0.870 |
| CSP log-variance features | correct | correct | 0.996 |

Aggregate power descriptors are largely blind to a class difference carried by
the **spatial covariance structure** — which is precisely what CSP extracts.
They can rank the wrong band, and a small τ then amplifies that error into a
confident peak.

Two consequences for the manuscript's own figures:

- Fig. 4's subject-averaged weights span 0.11–0.27; the manuscript-style
  descriptor at τ = 0.1 reproduces this almost exactly (0.179–0.26). Near-uniform
  weights are the expected output of this descriptor, not a property of the data.
- Eq. (18)'s peaked vector (0.6339 on one band) is reproduced at τ = 0.02 — **on
  the wrong band, at 0.575**. A dominant band in the weight vector is therefore
  not evidence of genuine band specificity.

**Recommendation:** compute I_b on the band's CSP log-variance features, inside
the training fold, and report a τ (or γ) sensitivity curve. Report the weight
entropy H(w) = −Σ w_b log w_b alongside the weights, so readers can see how far
from uniform the solution actually is.

---

## 4. When should this be expected to help?

(M1) can only help when a subject's discriminative information really is
concentrated in a subset of bands. That is a testable prediction, and it is the
mechanism claim the rebuilt paper should be built on:

> **H1.** The per-subject gain Δ_i = Acc_i(M1) − Acc_i(FBCSP) increases as the
> weight distribution departs from uniform, i.e. Δ_i correlates negatively with
> H(w_i).

If H1 holds, near-uniform weights and near-zero average gain are no longer an
embarrassment — they are the predicted behaviour of a method that only acts when
there is something to act on, and Fig. 4 becomes supporting evidence. If H1
fails, the MI prior carries no usable information and the approach should be
abandoned rather than tuned.

This is experiment **E10**. It should be run before E1.

---

## 5. Statement of the rebuilt method

1. Band-pass into B sub-bands; per band, fit CSP on the training fold and take
   2m log-variance features.
2. Per band, estimate I_b = I(F^(b); y) on the **training fold**, using a kNN
   estimator with stated k, on the **CSP features**.
3. w = softmax(I/τ). Report w, H(w), and the τ sensitivity curve.
4. Standardise F on the training fold; rescale block b by w_b^{γ/2}.
5. Fit an L2-penalised linear classifier. Report λ, γ, and recover β_b =
   w_b^{γ/2} α_b for interpretation.

Steps 4–5 together solve (M1). Every quantity that changes the result is named
and reported.

---

## References to add

- Zou (2006), *The adaptive lasso and its oracle properties*, JASA — the oracle
  property that (M1) inherits
- Ang et al. (2008, 2012) — FBCSP and MIBIF, the direct predecessor
- Zhang et al. — SFBCSP; Zhang et al. — SBLFB, the sparse alternatives
- Mane et al. (2021), *FBCNet* — filter bank + spatial convolution + variance
  layer, the closest deep competitor and the one currently missing
- Ledoit & Wolf (2004) — the shrinkage estimator whose internal standardisation
  makes shrinkage-LDA scale-invariant
