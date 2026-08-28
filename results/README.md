# Result files

One row per subject. Written incrementally by the runners, so a file is valid
even if a run was interrupted.

| file | produced by | rows |
|---|---|---|
| `descriptors.csv` | `run_descriptors.py` | 211 |
| `descriptors_fixedbands.csv` | `run_descriptors.py --bands fixed` | 211 |
| `sweep.csv` | `run_sweep.py` (aggregate descriptor) | 209 |
| `sweep_csp.csv` | `run_sweep.py --descriptor csp` | 211 |
| `sweep_csp_fixedbands.csv` | `run_sweep.py --descriptor csp --bands fixed` | 211 |
| `split_control.csv` | `split_control.py` | 211 |
| `fbcnet.csv` | `run_fbcnet.py` | 211 |

`sweep.csv` has 209 rows rather than 211 because two subjects failed the
minimum-trials-per-class check under that configuration.

Copy the CSV files from your run into this directory before publishing the
repository.
