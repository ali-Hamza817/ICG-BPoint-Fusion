# ICG B-point project notes (resume file)

Goal: apply Sheikh et al. 2022 (AdaBoost + 465 features + variable search window) to the public
beat-to-beat data of Abel et al. 2026 (EmpkinS + Guardian), compare to Abel's RF (MAE 8.13 ms),
and look for improvements (target: cross-dataset generalisation).

## Hard rules (from user)
- Work ONLY inside this folder. Never delete anything (new files only; deletes need user approval).
- No sudo / admin rights.
- NEVER report fake or unverified numbers. Every number in reports must come from a logged run.
- Pipelines run detached (setsid nohup) and must be resumable.

## Layout
- data/                raw datasets (gitignored), downloaded by download_datasets.sh (done, md5 verified)
- repos_pepbench/      reference clone of github.com/empkins/pepbench (read only, NOT executed/installed)
- venv/                python venv (pip packages from PyPI only)
- pipeline/            our code (numbered steps), outputs in results/, figures in figures/
- logs/                run logs

## Data facts (verified from files)
- Labels: rater_01 / rater_02 csv per phase: heartbeat start/end + B-point (ICG) / Q-wave onset (ECG) samples.
- Empkins: 1000 Hz, `sample_absolute` indexes the FULL .acq recording.
- Guardian: 500 Hz, `sample_absolute` indexes the per-phase array in the .mat (RAW_SIGNALS split by IV event names).

## Status log
(append below, newest last)

### 2026-11-18 step 1 done: beats built (pipeline/01_build_beats.py -> data/processed/)
- Env: use `./venv_sys/bin/python` (system-site-packages venv + bioread + neurokit2). `venv/` is an unused empty venv (not deleted).
- 12,014 beats (EmpkinS 5,095; Guardian 6,919); B rater1 n=12,013; rater2 matched n=11,857 (rater-2 matched by position, ids differ).
- Validation vs Abel (measured): inter-rater |B1-B2| MAE 8.73+-17.0 ms (Abel 8.35+-14.78); Guardian PEP(B-Q) 138.4+-27.8 (Abel 138.4+-27.1).
- EmpkinS outliers: 103 beats with PEP<20 or >250 ms, 65 of them VP_002 tsst/talk (B label ~360 ms before R) -> annotation artefact.
  DECISION: analysis cohort = B1 present AND 20<=PEP(B1-Q1)<=250 ms; always also state raw counts. Guardian beats count differs from Abel (6,919 vs 6,611): their exclusion rules not replicated.
- Signal window per beat: 1 kHz grid, R-100..R+600 ms (Guardian upsampled from 500 Hz by cubic spline). R = argmax cleaned ECG in [Q-10,Q+80] ms.

### 2026-11-18 steps 2-4
- EDA done: 14 figures in figures/eda/, numbers in results/eda_summary.json. Cohort 11,903 beats (EmpkinS 4,990; Guardian 6,913); cohort PEP: EmpkinS 88.4+-25.3, Guardian 138.4+-27.6 (Abel: 88.4+-25.0 / 138.4+-27.1).
- 03: Sheikh 465 features (pipeline/03_sheikh_features.py, ASSUMPTIONS marked in code); 03b: own rule-based B estimates (not PEPbench code).
- 04_models.py: 520 resumable tasks -> results/tasks/*.pkl (files prefixed SMOKE__ are smoke tests, ignore). Launch/resume: `./run_all.sh` (detached). AdaBoost tuned on 4-config SUBSET of Sheikh grid (full grid infeasible, ~1 s/tree).
- NEXT: pipeline/05_report.py (metrics, tables, figures) reading results/tasks only.
