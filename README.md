# ICG-BPoint-Fusion

Applying the B-point detection approach of Sheikh et al. (2022, *Psychophysiology*) to the public beat-to-beat datasets of Abel et al. (2026, *Front. Digit. Health*), and testing whether combining both feature families improves accuracy and cross-dataset generalisation.

> Status: work in progress. Results are preliminary; see `reports/RESULTS_SUMMARY.md` for caveats.

## Where things are

| Folder / file | Content |
|---|---|
| `reports/EDA_REPORT.md` | Full EDA report (numbers + all figures) |
| `reports/RESULTS_SUMMARY.md` | Model results tables and caveats |
| `figures/eda/` | 14 EDA figures |
| `figures/results/` | Model ranking, cross-dataset ranking, best-model diagnostics |
| `results/` | `eda_summary.json`, `table_*.csv`, `reference_numbers.json`, `tasks/` (saved predictions of all 520 model runs) |
| `pipeline/` | Numbered scripts `01`…`07` (build beats, EDA, features, models, reports) and `common.py` |
| `run_all.sh` | Resumable end-to-end pipeline (skips finished steps) |
| `download_datasets.sh` | Resumable, checksum-verified download of EmpkinS and Guardian from OSF |
| `logs/` | Run logs |
| `external/` | Local clone of PEPbench used for reference only (not committed) |
| `data/` | Raw + processed data (not committed; regenerate with the two scripts above) |
| `*.pdf` | The two papers |

## Reproduce

```bash
./download_datasets.sh          # ~2.3 GB, OSF (sh3xn EmpkinS, gyh75 Guardian)
python3 -m venv --system-site-packages venv_sys && venv_sys/bin/pip install bioread neurokit2
# needs numpy pandas scipy scikit-learn lightgbm matplotlib seaborn shap joblib tabulate
./run_all.sh                    # EDA + features + 520 model tasks (~6 h on a shared 80-thread server)
cd pipeline && ../venv_sys/bin/python 06_eda_report.py && ../venv_sys/bin/python 07_results_summary.py
```

## Data

EmpkinS (OSF `sh3xn`) and Guardian (OSF `gyh75`), published with PEPbench (Richer et al., 2025). Sheikh's Emory data are confidential and not used. Labels used: rater 1; rater 2 only for agreement analysis.

## Method notes

- Own loaders (PEPbench code is not executed). Beats are R-aligned 1 kHz windows (R−100 … R+600 ms); ICG band-passed 0.5–40 Hz.
- Cohort: cycles with a rater-1 B-point and PEP (B − Q) between 20 and 250 ms.
- Sheikh features: 465-dimensional vector re-implemented from the paper's Appendix B (assumptions marked `ASSUMPTION` in `pipeline/03_sheikh_features.py`).
- Validation: participant-level folds (5 folds x 3 repeats), inner 3-fold group CV for configuration choice; EmpkinS to Guardian transfer in both directions.
