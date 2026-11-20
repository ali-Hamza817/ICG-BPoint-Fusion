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

### 2026-11-19 fair-evaluation phase (steps 08-10)
- PEPbench/biopsykit install FAILED (PyPI ~2 KB/s, ReadTimeout x3). venv_pepbench/ is an unused partial venv (gitignored, not deleted). Fallback: own re-implementation of Abel Table-1 rules (pipeline/08_abel_features.py, 11 of 12; Miljkovic omitted; assumptions marked in code). Single-rule MAE vs rater1 in results/table_abel11_rule_mae.csv.
- 09_fair_models.py: 963 resumable tasks -> results/tasks_fair/ (results/tasks_fair_SMOKE = smoke test, ignore). Pooled CV x3 repeats; cross-dataset candidates chosen by SOURCE-only CV (srccv_* tasks); per-subject z-score variants (_sz).
- 10_fair_report.py -> results/fair_table_*.csv, reports/FAIR_RESULTS.md, figures/results/fair_pooled_and_cross.png.
- Resume/launch (detached): `setsid nohup ./run_fair.sh > logs/run_fair.log 2>&1 < /dev/null &`

### 2026-11-19 ADVISOR DIRECTION (Sheikh): per-patient beats + feature comparison first
- Fair-evaluation run (run_fair.sh / 09_fair_models.py) PAUSED by user request after 30/963 tasks (results/tasks_fair/, resumable: `setsid nohup ./run_fair.sh > logs/run_fair.log 2>&1 < /dev/null &`).
- Task 1 DONE: pipeline/11_beat_table.py -> dataset_beats/ (beats_all_patients.csv, per_patient/*.csv, waveforms/*.npz, patient_summary.csv) + figures/per_patient/*.png (39 patients, 12,014 beats, 11,903 usable).
- Task 2 DONE: reports/FEATURE_COMPARISON.md (Abel 13 features vs Sheikh 465).
- Advisor premises checked vs papers: Abel DID use 5-fold nested subject-wise CV + cross-dataset; Sheikh Study A = 140 men age 68+-2.5, Study B = 49 pts age 35+-12.7 (so not all old); Abel ages 23.1+-2.6 / 31.2+-11.0.
- TODO: fix For18 in 08_abel_features.py (Abel Table 3: defined on d3Z/dt3); single-beat vs ensemble-averaged test; morphology-only ablation.
- 2026-11-19 CODE DOWNLOADED: external/abel_algorithms_biopsykit/ (Abel's 12 B-point algorithms + trained-model code, raw files from mad-lab-fau/BioPsyKit via GitHub API), external/sheikh_toolbox_B_point/ (Sheikh feat_vec_465.m, AdaBoost notebook, Noise_removal.m = his ensemble averaging, demo data). external/sheikh_ICG_OSToolbox + external/biopsykit_source = partial git clones (killed, too slow; not deleted, ignore).
- Sheikh's REAL code findings: StandardScaler before AdaBoost; featureless beats (no notch & no inflection) predicted by rule RB_max_2D, not the model; tuned AdaBoost hp mostly max_features 0.7, depth 12-22, lr 0.01-1 (outside my old 4-config grid); window upper limit = first sample above 0.66*C; 453 amplitude = ICG value at max-d3 point (old port used d3 value).
- pipeline/03c_sheikh_features_v2.py = line-by-line port; 12_validate_sheikh_port.py: exact match (0.0 dev) on his 2 demo beats (inflection-found branch NOT exercised by demo).

### 2026-11-19 advisor directions 1-3 (v3 pipeline)
- Dir 1 DONE: external/biopsykit_shim/ = Abel's 12 B-point algorithm files UNMODIFIED + tiny shim (tpcp/validators). pipeline/13_abel_real_features.py -> data/processed/abel_real_X.csv, results/abel_real_rule_mae.csv. Findings: Abel's C extractor (prominence=1) fails on 10.1% Guardian beats (low-amplitude subjects), 0.2% EmpkinS. Pale2021 MAE ~100+ ms (unit-dependent slope threshold; Abel's own code).
- Dir 2: pipeline/14_models_v3.py (resumable, results/tasks_v3/, 184 tasks, `./run_v3.sh` detached) = Abel replica RF + Sheikh-way AdaBoost (StandardScaler, his hp region, featureless rule) + fusion + cheap models; 15_report_v3.py -> reports/V3_RESULTS.md. AdaBoost cost ~2 s/tree (500 trees ~16 min/fit).
- Dir 3: pipeline/16_ensemble_beats.py <window_s> (port of Noise_removal.m) -> data/processed/ea_*_w{10,20,30}. NOTE annotated beats per phase span only ~30 s (~40 beats) so 60 s windows impossible; used 10/20/30 s. EA label = mean rater-1 RB of surviving beats (ASSUMPTION). EA beats: w10 469 (80 Emp/389 Gdn), w20 258 (46/212), w30 106 (0/106). His amplitude thresholds delete 41% of EmpkinS beats (scale mismatch). 17_ea_vs_single.py <w> compares EA-model vs single->avg.
- 2026-11-19 v3 + EA RESULTS DONE (all 184 tasks; ea_vs_single w10/w20). Summary for advisor: reports/ADVISOR_SUMMARY.md; full tables reports/V3_RESULTS.md; results/v3_table_*.csv, results/ea_vs_single_w*.csv. Headline: Sheikh465+AdaBoost 7.02 ms vs Abel replica RF 10.17 ms (replica does NOT reproduce Abel's published 8.13). Cross-dataset E->G 13.0 (Abel pub. best 19.39); G->E 20.0 (Abel pub. SVR 12.70 NOT beaten). Nothing committed/pushed since 6bc8284 (awaiting user OK; commits must carry no Claude co-author line).

### 2026-11-20 TASK 2 (advisor): dataset ECG+ICG+B1+PEP1+C per beat -> Sheikh feat_vec_465 -> plots
- DONE: pipeline/19_task2_dataset.py -> dataset_task2/ (task2_beats.csv, signals/*.npz, matlab_demo_format_{EmpkinS,Guardian}.zip, features_465.npy/index/columns, README.md; 104 MB). pipeline/20_task2_plots.py -> figures/task2/ (dataset_overview, example_beats, features_465_summary, strips/*.png 39).
- Facts: 11,614 beats usable by his .m (0<=RB<=150); 289 usable beats (RB>150 or <0) cannot be processed by his MATLAB file (feat_icg(RB_out+1) beyond 151 samples). His-format route == window route (dev 0). No MATLAB/Octave on server: his .m not executed here.
- Pages viewer (task 1) live: https://renofusion.me/ICG-BPoint-Fusion/ (docs/). Task 2 outputs NOT committed yet (awaiting user OK; dataset_task2 = 104 MB).
