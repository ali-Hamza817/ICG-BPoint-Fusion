#!/usr/bin/env bash
# Resumable v3 chain: real Abel features (13) + real-Sheikh features (3c) + models (14) + report (15, if present).
# Run detached:  setsid nohup ./run_v3.sh > logs/run_v3.log 2>&1 < /dev/null &
cd "$(dirname "$0")/pipeline" || exit 1
PY=../venv_sys/bin/python
[ -f ../data/processed/abel_real_X.csv ]   || $PY -u -W ignore 13_abel_real_features.py   > ../logs/13_abel_real_features.log 2>&1
[ -f ../data/processed/sheikh_v2_X.npy ]   || $PY -u -W ignore 03c_sheikh_features_v2.py  > ../logs/03c_sheikh_v2.log 2>&1
$PY -u -W ignore 14_models_v3.py 40 >> ../logs/14_models_v3.log 2>&1
[ -f 15_report_v3.py ] && $PY -u -W ignore 15_report_v3.py >> ../logs/15_report_v3.log 2>&1
echo "V3 PIPELINE FINISHED $(date)"
