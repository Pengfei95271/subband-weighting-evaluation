# Sub-band weighting in filter-bank CSP: evaluation code

Code and results for *Mutual-information-guided sub-band weighting does not
improve filter-bank CSP: evidence from 211 subjects across five datasets*.

Every number in the manuscript traces to a file here. The table below maps
manuscript sections to the script that produced them and the result file that
holds them.

---

## Quickest check

Two scripts confirm the central claims without downloading any data. Both take
under a minute.

```bash
pip install numpy scipy scikit-learn pandas matplotlib
python scale_invariance.py     # §3.1 — the weighting is inert
python verify_weights.py       # §4.2 — what the weight formula can produce
```

Expected values are in `VALIDATION_reference.md`.

## Full install

```bash
pip install -r requirements.txt          # adds moabb, mne, pyriemann, torch
python test_synthetic.py                 # 22/22 expected
python test_fbcnet.py                    # 9/9 expected
```

Datasets download automatically through MOABB on first use. Budget 200 GB;
Lee2019-MI alone is about 61 GB.

---

## Manuscript section → script → result file

| section | claim | script | result file |
|---|---|---|---|
| §3.1 | weighting is inert; 6 of 9 classifiers scale-invariant | `scale_invariance.py`, `run_sweep.py` | `results/sweep.csv` |
| §3.2 | descriptor accuracy against a behavioural reference | `run_descriptors.py` | `results/descriptors.csv` |
| §3.3 | γ sweep, both descriptors | `run_sweep.py --descriptor {logvar_li,csp}` | `results/sweep.csv`, `results/sweep_csp.csv` |
| §3.4 | split-half control | `split_control.py` | `results/split_control.csv` |
| §3.5 | FBCNet baseline | `run_fbcnet.py` | `results/fbcnet.csv` |
| §3.6 | spread modulates the gain | derived from `descriptors.csv` + `sweep_csp.csv` | — |
| §3.7 | seven-band replication | both runners with `--bands fixed` | `results/*_fixedbands.csv` |
| §4.2 | what the weight formula can produce | `verify_weights.py` | — |
| Fig 1 | schematic | `make_figure1.py` | — |
| Fig 2 | descriptor accuracy by decoding level | `make_figure2.py` | reads `descriptors.csv` |
| Fig 3 | γ sweep and per-dataset effects | `make_figure3.py` | reads both sweep files |
| Table 1 | classifier invariance | `scale_invariance.py` | — |

`fill_numbers.py` extracts the dataset properties, parameter counts and
training-fold sizes quoted in §2.1, §3.5 and §4.3.

---

## Reproducing from scratch

Long runs are resumable: results append per subject, and a restart skips what is
already done. Interrupt freely.

```bash
python run_descriptors.py --stage 1                          # ~3 h
python run_sweep.py --stage 1 --descriptor logvar_li         # ~2.5 h
python run_sweep.py --stage 1 --descriptor csp               # ~2.5 h
python split_control.py --stage 1                            # ~4 h
python run_fbcnet.py --stage 1 --max-epochs 200              # ~7 h, CPU
python run_descriptors.py --stage 1 --bands fixed            # ~3 h
python run_sweep.py --stage 1 --bands fixed --descriptor csp # ~3 h
```

Add `--analyse-only` to any runner to re-print the statistics from an existing
result file without recomputing.

The `results/` directory already contains the seven CSV files these commands
produce, so the analysis can be checked without running anything.

---

## Fixed settings

Held constant across every subject and fold. Nothing is tuned per subject.

| parameter | value |
|---|---|
| `n_components` (CSP filter pairs) | 4 |
| `tau` | 0.015 |
| `gamma` | swept over {0, 0.5, 1, 2, 4}, reported in full |
| covariance shrinkage | Ledoit–Wolf |
| MI estimator | `mutual_info_classif`, k = 3 |
| `mibif_k` | 4 |
| resample | 250 Hz |
| bands (published) | 8–12, 12–16, 16–20, 20–24, 24–30 Hz |
| bands (corrected, §3.7) | seven 4 Hz bands, 4–32 Hz |
| random seed | 42 |
| FBCNet | m = 32, 4 strides, 300 epochs max, patience 50 |

FBCNet's early-stopping values were fixed once on a single subject and then held
constant. A tighter patience of 20 under-trains the baseline by roughly five
points at this data scale, which is why it is stated rather than left to a
default.

---

## Leakage controls

Every data-dependent operation is confined to the training portion of each fold:
band-pass filtering, CSP estimation, mutual-information scoring, band weights,
feature selection, standardisation, and network training. The top-2 bands used
for the cross-band interaction feature are chosen from training-fold weights
only.

Analysis rules for the split-half control were fixed before it was run. They are
in `split_control.py`, and they decided the outcome: the full cohort gave a
separation of 1.99 points (p = 0.018), the trial-sufficient subset 1.23 points
(p = 0.181), and the pre-specified rule selects the latter.

---

## Datasets

All accessed through MOABB. 211 subjects, 3 to 64 channels.

| dataset | subjects | channels | sessions | trials/subject |
|---|---|---|---|---|
| BNCI2014-001 (BCI IV-2a) | 9 | 22 | 2 | 288 |
| BNCI2014-004 (BCI IV-2b) | 9 | 3 bipolar | 5 | 720 |
| Cho2017 | 52 | 64 | 1 | 200 |
| Lee2019-MI (OpenBMI) | 54 | 62 | 2 | 100 |
| Dreyer2023 | 87 | 27 | 1 | 240 |

---

## Notes for anyone reusing this

`awfbcsp/features.py` and `awfbcsp/pipelines.py` depend only on numpy, scipy and
scikit-learn, so the feature stack can be tested without MOABB.
`test_synthetic.py` exercises it on synthetic data with a known informative band.

The `AWFBCSP` estimator takes a `variant` argument with four settings. Two form a
matched pair: `published` applies the √w rescaling followed by standardisation,
`noweight` omits the rescaling. They should produce identical output, and
`diagnostics.weighting_is_effective()` measures the difference. If you are
implementing a weighting scheme of your own, that comparison is worth running
before attributing any gain to it.

`FBCSP-L2` is the matched control for `AWFBCSP-reg`: the same pipeline at γ = 0,
with the same classifier. Comparing `AWFBCSP-reg` against `FBCSP` instead would
confound the weighting with the change of estimator.

---

## Citation

**[to add on acceptance]**

## Licence

MIT.
