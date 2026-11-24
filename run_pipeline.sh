#!/usr/bin/env bash
# End-to-end run, from raw data to the final tables. Each step is resumable; the model step (14) is the heavy one (hours).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-venv_sys/bin/python}
./download_datasets.sh
cd pipeline
$PY 01_build_beats.py && $PY 22_artefact_flags.py
$PY 03_sheikh_features.py && $PY 03c_sheikh_features_v2.py && $PY 13_abel_real_features.py
TASKS_TAG=v4 $PY 14_models_v3.py 40
$PY 23_ada_cross_all_configs.py && $PY 24_final_report.py
$PY 28_audit.py                       # independent recomputation of every reported MAE
$PY 27_reproduce_headline.py          # retrain the headline models from scratch and compare predictions
