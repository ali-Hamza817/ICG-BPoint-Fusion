#!/usr/bin/env bash
# Re-run everything that depends on the usable-beat definition after the artefact fix (step 22). Resumable (14 skips finished tasks). Detached:
#   setsid nohup ./run_v4.sh > logs/run_v4.log 2>&1 < /dev/null &
cd "$(dirname "$0")/pipeline" || exit 1
PY=../venv_sys/bin/python; export TASKS_TAG=v4
$PY -u -W ignore 03_sheikh_features.py   > ../logs/v4_03_features.log 2>&1
$PY -u -W ignore 03c_sheikh_features_v2.py > ../logs/v4_03c.log 2>&1
$PY -u -W ignore 13_abel_real_features.py  > ../logs/v4_13.log 2>&1
$PY -u -W ignore 14_models_v3.py 40 >> ../logs/v4_14_models.log 2>&1
$PY -u -W ignore 15_report_v3.py   > ../logs/v4_15_report.log 2>&1
for w in 10 20; do $PY -u -W ignore 16_ensemble_beats.py $w > ../logs/v4_16_w$w.log 2>&1; $PY -u -W ignore 17_ea_vs_single.py $w > ../logs/v4_17_w$w.log 2>&1; done
echo "V4 PIPELINE FINISHED $(date)"
