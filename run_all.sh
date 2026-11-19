#!/usr/bin/env bash
# Resumable full pipeline. Safe to re-run any time: finished steps/tasks are skipped. Run detached:
#   setsid nohup ./run_all.sh > logs/run_all.log 2>&1 < /dev/null &
cd "$(dirname "$0")/pipeline" || exit 1
PY=../venv_sys/bin/python
[ -f ../data/processed/beats_meta.csv ]  || $PY -W ignore 01_build_beats.py     > ../logs/01_build_beats.log 2>&1
[ -f ../results/eda_summary.json ]       || $PY -W ignore 02_eda.py              > ../logs/02_eda.log 2>&1
[ -f ../data/processed/sheikh_X.npy ]    || $PY -W ignore 03_sheikh_features.py  > ../logs/03_features.log 2>&1
[ -f ../data/processed/rule_X.csv ]      || $PY -W ignore 03b_rule_features.py   > ../logs/03b_rules.log 2>&1
$PY -u -W ignore 04_models.py 64 >> ../logs/04_models.log 2>&1
[ -f 05_report.py ] && $PY -u -W ignore 05_report.py >> ../logs/05_report.log 2>&1
echo "PIPELINE FINISHED $(date)"
