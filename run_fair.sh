#!/usr/bin/env bash
# Resumable "fair evaluation" chain (steps 08-10). Safe to re-run; finished outputs/tasks are skipped. Run detached:
#   setsid nohup ./run_fair.sh > logs/run_fair.log 2>&1 < /dev/null &
cd "$(dirname "$0")/pipeline" || exit 1
PY=../venv_sys/bin/python
[ -f ../data/processed/abel11_X.csv ] || $PY -u -W ignore 08_abel_features.py > ../logs/08_abel_features.log 2>&1
$PY -u -W ignore 09_fair_models.py 48 >> ../logs/09_fair_models.log 2>&1
[ -f 10_fair_report.py ] && $PY -u -W ignore 10_fair_report.py >> ../logs/10_fair_report.log 2>&1
echo "FAIR PIPELINE FINISHED $(date)"
