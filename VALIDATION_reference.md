# Validation reference

For independent confirmation of the numerical claims in §3.1 and §4.2. Run the
two scripts and compare against the values below. Both are deterministic; the
seed is fixed and no data download is required for what is listed here.

    python scale_invariance.py
    python verify_weights.py

A third run, `python verify_weights.py --with-data`, needs MOABB and
BNCI2014_001; expected values for it are at the end.

Generated with numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0. Small differences in the
final digits across BLAS implementations are expected and immaterial — the
quantities below are round-off in magnitude, and what matters is the order of
magnitude, not the digits.

---

## What each script establishes

`scale_invariance.py` — that the published weighting is removed before any
classifier sees it, and that most classifiers could not see it in any case.

`verify_weights.py` — what weight vectors the released weight computation can
produce, and under which settings.

---

## Values to check

### scale_invariance.py

| quantity | expected |
|---|---|
| relative Frobenius difference after z-score | 3.7 × 10⁻¹⁶ |
| classifiers invariant without standardisation | 7 of 10 tested |
| unpenalised linear model, max abs difference in p | ~1 × 10⁻⁷ |
| adaptive group ridge (M1) vs rescaled design (M2) | 1.1 × 10⁻¹⁶ |

Invariant: LDA (both solvers), Gaussian naive Bayes, decision tree, random
forest, gradient boosting, AdaBoost. Scale-sensitive: logistic regression,
SVM (RBF), k-nearest neighbours. Of the nine classifiers used in the study
under discussion, six are invariant.

### verify_weights.py (sections A and C)

| quantity | expected |
|---|---|
| analytic ceiling on the maximum weight, B = 5 | 0.6488 |
| temperature reproducing the published weight vector | τ = 0.5, L1 error 1 × 10⁻⁴ |
| bands that must share the lowest raw MI to reach it | 2 of 5 |

### verify_weights.py --with-data (nine IV-2a subjects)

| quantity | expected |
|---|---|
| raw MI range across bands | 0.00 – 0.08 nats (ceiling ln 2 = 0.693) |
| subjects with a band truncated to exactly 0 | 5 of 9 |
| maximum weight, across subjects | 0.465 – 0.647, mean 0.531 |
| subjects reaching the published 0.6339 | 1 of 9 |

---

## Full output as generated

### scale_invariance.py

```
========================================================================
Evidence for METHOD.md
  n=400 trials, p=40 features, B=5 bands, w=[0.05, 0.08, 0.62, 0.15, 0.1]
========================================================================

[1] Per-feature z-score cancels per-feature scaling (Eq. 10 + Alg. 2)
    relative Frobenius difference : 3.736e-16
    -> the weighting is removed before any classifier sees it

[2] Without the z-score: which classifiers can even see the weights?
    classifier                     behaviour                           
    ------------------------------------------------------------------
    LDA (lsqr, shrinkage=auto)     INVARIANT - weighting has no effect 
    LDA (svd, no shrinkage)        INVARIANT - weighting has no effect 
    Naive Bayes                    INVARIANT - weighting has no effect 
    Decision Tree                  INVARIANT - weighting has no effect 
    Random Forest                  INVARIANT - weighting has no effect 
    Gradient Boosting              INVARIANT - weighting has no effect 
    AdaBoost                       INVARIANT - weighting has no effect 
    Logistic Regression (C=1)      scale-sensitive                     
    SVM (RBF)                      scale-sensitive                     
    KNN                            scale-sensitive                     
    -> 7/10 classifiers are structurally invariant

[3] Why shrinkage-LDA is invariant: scikit-learn standardises internally
      sc = StandardScaler()  # standardize features
      X = sc.fit_transform(X)
      s = ledoit_wolf(X)[0]
      s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]

[4] An unpenalised linear model is a reparameterisation, not a change
    max |delta p| = 1.287e-07
    -> the downstream penalty is what makes weighting act (see METHOD 2.1)

[5] Adaptive group ridge (M1) == plain L2 on the rescaled design (M2)
    max |beta_direct - beta_reparameterised| = 1.110e-16
    equivalence holds: True

    coefficient mass per band (gamma=1.0):
      band        w_b   ||beta_b|| (M1)   ||beta_b|| (ridge)
      0         0.050            0.0836               0.0929
      1         0.080            0.0409               0.0437
      2         0.620            0.1786               0.1797
      3         0.150            0.1400               0.1456
      4         0.100            0.1746               0.1840

========================================================================
SUMMARY
========================================================================
  z-score cancels the weighting          : True  (3.7e-16)
  classifiers blind to it even without   : 7/10
  (M1) == (M2) to machine precision      : True  (1.1e-16)

  The published mechanism cannot act. The regularisation form can.
========================================================================
```

### verify_weights.py

```
==================================================================
A. Analytic ceiling of the released formula
==================================================================
   min-max forces max(score)=1 and min(score)=0, so the most
   peaked weight vector possible is one band at 1, rest at 0.

   B        ceiling        uniform       
   --------------------------------------
   3        0.7870         0.3333        
   5        0.6488         0.2000        
   7        0.5519         0.1429        
   9        0.4802         0.1111        

   published max weight : 0.6339
   ceiling at B=5       : 0.6488
   reachable            : True
   -> within 0.0149 of the ceiling; see section [c2] for how many
      bands must share the lowest raw MI to reach it.

==================================================================
C. What settings would reproduce Eq. (18)?
==================================================================

   [c1] With min-max normalisation, which tau gets closest?
        tau      max_w      L1 err     weights                     
        --------------------------------------------------------------
        2.0      0.2886     0.6905     [0.289, 0.182, 0.179, 0.175, 0.175]
        1.0      0.3970     0.4738     [0.397, 0.158, 0.153, 0.146, 0.146]
        0.5      0.6339     0.0001     [0.634, 0.1, 0.095, 0.086, 0.086]  <- code default
        0.3      0.8626     0.4574     [0.863, 0.04, 0.036, 0.031, 0.031]
        0.2      0.9691     0.6703     [0.969, 0.01, 0.008, 0.007, 0.007]
        0.1      0.9997     0.7317     [1.0, 0.0, 0.0, 0.0, 0.0]   
        0.05     1.0000     0.7322     [1.0, 0.0, 0.0, 0.0, 0.0]   
        0.02     1.0000     0.7322     [1.0, 0.0, 0.0, 0.0, 0.0]   

        closest tau = 0.5 (L1 error 0.0001)

   [c2] What raw MI spread would the code need at tau=0.5?
        implied normalised scores : [0.0, 0.0488, 0.9999, 0.0761, 0.0]
        max of those              : 0.9999
        -> requires 2 of 5 bands to share the identical raw MI

   [c3] Without the min-max step (paper's Eq. 8-9 as written):
        tau=0.1    implied raw MI spread = 0.2000 nats  
        tau=0.05   implied raw MI spread = 0.1000 nats  
        tau=0.02   implied raw MI spread = 0.0400 nats  
        tau=0.01   implied raw MI spread = 0.0200 nats  

==================================================================
```
