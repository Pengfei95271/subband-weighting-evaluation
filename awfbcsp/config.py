"""
Datasets, frequency bands, and protocol configuration.

Frequency bands
---------------
The manuscript's bank is 8-12 / 12-16 / 16-20 / 20-24 / 24-30 Hz with a 4-30 Hz
band-pass, which leaves 4-8 Hz filtered in but belonging to no sub-band, makes
the last band 6 Hz wide with no stated reason, and labels 24-30 Hz as gamma when
it is high beta (issue P1-2). BANDS_FIXED closes the gap and keeps every band
4 Hz wide. BANDS_PAPER is kept so the original can be reproduced exactly.
"""

# (low, high, label)
BANDS_PAPER = [(8, 12, "mu"), (12, 16, "low beta"), (16, 20, "beta"),
               (20, 24, "high beta"), (24, 30, "high beta 2")]

BANDS_FIXED = [(4, 8, "theta"), (8, 12, "mu"), (12, 16, "low beta"),
               (16, 20, "beta"), (20, 24, "high beta"),
               (24, 28, "high beta 2"), (28, 32, "low gamma")]


def band_edges(bands):
    return [[lo, hi] for lo, hi, _ in bands]


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
# 'sessions' is what the data actually contains, not what the manuscript claims.
# BNCI2014_004 (IV-2b) has 5 sessions per subject, not 2 (issue P1-6).

DATASETS = {
    "BNCI2014_001": dict(  # BCI IV-2a
        subjects=9, channels=22, sessions=2, role="development",
        protocols=("within", "cross_session", "cross_subject")),
    "BNCI2014_004": dict(  # BCI IV-2b
        subjects=9, channels=3, sessions=5, role="low-channel / non-stationarity",
        protocols=("within", "cross_session", "cross_subject")),
    "Lee2019_MI": dict(
        subjects=54, channels=62, sessions=2, role="main / channel subsampling",
        protocols=("within", "cross_session", "cross_subject")),
    "Cho2017": dict(
        subjects=52, channels=64, sessions=1, role="replication / learning curve",
        protocols=("within", "cross_subject")),
    "Dreyer2023": dict(
        subjects=87, channels=None, sessions=1,
        role="large cross-subject / EMG screening / subject profiles",
        protocols=("within", "cross_subject")),
}

STAGE0 = ["BNCI2014_001", "BNCI2014_004"]          # two-week gate
STAGE1 = ["Lee2019_MI", "Cho2017", "Dreyer2023"]   # full run

DEFAULTS = dict(
    resample=250.0,      # align 512 Hz (Cho2017) and 1000 Hz (Lee2019) to 250 Hz
    fmin=4.0, fmax=32.0,
    n_components=4,      # filter PAIRS, so 8 features per band
    # tau=0.1 leaves the weights near-uniform on real data (max_w ~0.23 on
    # IV-2a S1, i.e. exactly Fig. 4). 0.015 is where they actually separate.
    tau=0.015,
    gamma=1.0,
    # k=8 scored identically to k=40 on IV-2a S1 -- shrinkage-LDA had already
    # suppressed the redundant dimensions, so the baseline was doing nothing.
    mibif_k=4,
    shrinkage="ledoit_wolf",
    random_state=42,
    n_seeds=5,
)


def load_dataset(name, subjects=None):
    """Instantiate a MOABB dataset by name, tolerating the 1.0 renaming."""
    import moabb.datasets as md

    obj = getattr(md, name, None)
    if obj is None:  # pre-1.0 names had no underscore
        obj = getattr(md, name.replace("_", ""), None)
    if obj is None:
        raise ImportError(
            f"{name} not found in this MOABB version. Check "
            "moabb.datasets.utils.dataset_list for the current roster.")
    return obj(subjects=subjects) if subjects else obj()
