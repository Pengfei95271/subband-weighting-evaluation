"""Go/No-Go gate for the AWFBCSP manuscript."""

from .config import BANDS_FIXED, BANDS_PAPER, DEFAULTS, STAGE0, STAGE1, band_edges
from .diagnostics import (
    add_relative_noise,
    noise_calibration,
    weight_summary,
    weighting_is_effective,
)
from .experiments import (
    compare_descriptors,
    e2_learning_curve,
    e3_ablation,
    e6_noise,
    e10_entropy_vs_gain,
    summarise_e10,
    sweep_gamma,
    sweep_tau,
)
from .features import CSP, FilterBankCSP, MIBandWeights, MIBIFSelect
from .pipelines import AWFBCSP, make_broadband_pipelines, make_pipelines
from .stats import (
    cd_diagram,
    compare_all,
    format_verdict,
    friedman_nemenyi,
    per_subject_scores,
    verdict,
)

__version__ = "0.1.0"

__all__ = [
    "AWFBCSP", "CSP", "FilterBankCSP", "MIBandWeights", "MIBIFSelect",
    "make_pipelines", "make_broadband_pipelines",
    "BANDS_PAPER", "BANDS_FIXED", "band_edges", "DEFAULTS", "STAGE0", "STAGE1",
    "weighting_is_effective", "noise_calibration", "add_relative_noise",
    "weight_summary",
    "per_subject_scores", "compare_all", "friedman_nemenyi", "cd_diagram",
    "verdict", "format_verdict",
    "e2_learning_curve", "e3_ablation", "e6_noise", "e10_entropy_vs_gain",
    "summarise_e10", "sweep_tau", "sweep_gamma", "compare_descriptors",
]
