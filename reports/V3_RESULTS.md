> **SUPERSEDED (2026-11-20).** The numbers in this report were computed before the artefact-label fix (`pipeline/22_artefact_flags.py`): 199 beats whose rater-1 B point was an "Artefact" marker were counted as usable. Corrected results: `reports/V4_RESULTS.md` (when present) and the README. Kept unchanged below for traceability.

# v3 results: real Abel features vs real-Sheikh features (auto-generated from saved predictions)

All numbers computed by `pipeline/15_report_v3.py` from `results/tasks_v3/`. Subject-wise grouped CV (5 folds; 3 repeats for fast models, 1 for AdaBoost); hyper-parameters tuned only by inner 3-fold group CV on the training part.

## Caveats (read first)
- Abel's 12 algorithms are Abel's own files run through a small shim (`external/biopsykit_shim`); deviations: heartbeats = annotated reference heartbeats with our R-peaks (not PEPbench's neurokit segmentation + matching).
- Abel's C-point extractor (fixed prominence 1) fails on low-amplitude beats: 10.1 percent of Guardian beats vs 0.2 percent of EmpkinS; beats where all 12 rules failed are excluded (as Abel does). Evaluation set here: 11195 beats (Abel: 11,239).
- Sheikh features are a line-by-line port of his MATLAB (validated exactly on his 2 demo beats). His AdaBoost was tuned on a small grid inside his published range (max_features 0.7, depth 12/20, lr 0.1/1; 100 trees inner, 500 trees final).
- Not directly comparable with the numbers published by Abel (different cohort/folds); a replica of his design on OUR data is the fair baseline.

## Pooled subject-wise CV (MAE ms vs rater 1)

| model           | description                             |     n |   MAE |   MAE_rep_sd |    SD |   MedAE |    ME |   subjectMAE |    r |   n_rep |
|:----------------|:----------------------------------------|------:|------:|-------------:|------:|--------:|------:|-------------:|-----:|--------:|
| sheikh_ada      | Sheikh 465, AdaBoost                    | 11195 |  7.01 |       nan    | 13.22 |    2    |  0.49 |         7.45 | 0.9  |       1 |
| fus_ada         | Fusion (465+13), AdaBoost               | 11195 |  7.04 |       nan    | 13.19 |    2    |  0.39 |         7.49 | 0.9  |       1 |
| fus_rf          | Fusion, RF                              | 11195 |  8.11 |         0.1  | 12.34 |    3.09 | -0.16 |         8.58 | 0.9  |       3 |
| sheikh_rf       | Sheikh 465, RF                          | 11195 |  8.16 |         0.16 | 12.53 |    3.04 | -0.23 |         8.63 | 0.9  |       3 |
| sheikh_ada+rule | Sheikh 465, AdaBoost + featureless rule | 11195 |  8.43 |       nan    | 15.01 |    2    |  1.3  |         8.69 | 0.87 |       1 |
| fus_ada+rule    | Fusion, AdaBoost + featureless rule     | 11195 |  8.46 |       nan    | 15.02 |    2    |  1.2  |         8.73 | 0.87 |       1 |
| fus_lgbm        | Fusion, LightGBM                        | 11195 |  8.52 |         0.09 | 12.1  |    3.82 |  0.09 |         9.01 | 0.9  |       3 |
| sheikh_lgbm     | Sheikh 465, LightGBM                    | 11195 |  8.64 |         0.16 | 12.29 |    3.84 | -0.07 |         9.1  | 0.9  |       3 |
| abel_rf         | Abel replica: 12 PEPbench rules+RR, RF  | 11195 | 10.17 |         0.15 | 14.54 |    4.46 |  0.29 |        10.78 | 0.86 |       3 |
| fus_svr         | Fusion, SVR                             | 11195 | 10.39 |         0.09 | 12.92 |    5.27 |  0.75 |        10.96 | 0.87 |       3 |
| sheikh_svr      | Sheikh 465, SVR                         | 11195 | 11    |         0.09 | 13.77 |    5.49 |  0.72 |        11.63 | 0.86 |       3 |
| abel_svr        | Abel rules, SVR (imputed)               | 11195 | 11.88 |         0.11 | 18.1  |    5.63 |  0.85 |        12.59 | 0.81 |       3 |
| fus_ridge       | Fusion, Ridge                           | 11195 | 13.03 |         0.1  | 13.2  |    9.02 |  0.11 |        13.64 | 0.84 |       3 |
| sheikh_ridge    | Sheikh 465, Ridge                       | 11195 | 13.93 |         0.06 | 13.81 |    9.91 | -0.04 |        14.61 | 0.82 |       3 |

## Paired comparison with the Abel replica (rep 0; per-subject MAE; negative = candidate better)

| candidate       | description                             |   n_subjects |   replica_subjectMAE |   candidate_subjectMAE |   mean_diff |   ci95_lo |   ci95_hi |   wilcoxon_p |   subjects_improved |
|:----------------|:----------------------------------------|-------------:|---------------------:|-----------------------:|------------:|----------:|----------:|-------------:|--------------------:|
| sheikh_ada      | Sheikh 465, AdaBoost                    |           39 |               10.632 |                  7.447 |      -3.185 |    -3.827 |    -2.53  |        0     |                  36 |
| fus_ada         | Fusion (465+13), AdaBoost               |           39 |               10.632 |                  7.488 |      -3.144 |    -3.732 |    -2.523 |        0     |                  37 |
| sheikh_rf       | Sheikh 465, RF                          |           39 |               10.632 |                  8.413 |      -2.219 |    -2.853 |    -1.624 |        0     |                  36 |
| fus_rf          | Fusion, RF                              |           39 |               10.632 |                  8.44  |      -2.192 |    -2.82  |    -1.607 |        0     |                  36 |
| sheikh_ada+rule | Sheikh 465, AdaBoost + featureless rule |           39 |               10.632 |                  8.693 |      -1.94  |    -2.912 |    -0.906 |        0     |                  30 |
| fus_ada+rule    | Fusion, AdaBoost + featureless rule     |           39 |               10.632 |                  8.733 |      -1.9   |    -2.807 |    -0.886 |        0     |                  31 |
| sheikh_lgbm     | Sheikh 465, LightGBM                    |           39 |               10.632 |                  8.895 |      -1.737 |    -2.393 |    -1.104 |        0     |                  33 |
| fus_lgbm        | Fusion, LightGBM                        |           39 |               10.632 |                  8.903 |      -1.73  |    -2.396 |    -1.106 |        0     |                  33 |
| fus_svr         | Fusion, SVR                             |           39 |               10.632 |                 10.861 |       0.228 |    -0.498 |     0.941 |        0.38  |                  16 |
| sheikh_svr      | Sheikh 465, SVR                         |           39 |               10.632 |                 11.578 |       0.946 |     0.177 |     1.762 |        0.002 |                  10 |
| abel_svr        | Abel rules, SVR (imputed)               |           39 |               10.632 |                 12.49  |       1.857 |     1.107 |     2.774 |        0     |                   4 |
| fus_ridge       | Fusion, Ridge                           |           39 |               10.632 |                 13.637 |       3.005 |     2.161 |     3.925 |        0     |                   2 |
| sheikh_ridge    | Sheikh 465, Ridge                       |           39 |               10.632 |                 14.522 |       3.89  |     3.024 |     4.817 |        0     |                   1 |

## Cross-dataset (train on source only, test on target; no selection on target)

| direction         | model           |    n |   MAE |   subjectMAE |   subjectMAE_sd |    SD |     ME | best_cfg                 |
|:------------------|:----------------|-----:|------:|-------------:|----------------:|------:|-------:|:-------------------------|
| EmpkinS->Guardian | fus_ada         | 6215 | 13    |        13.73 |            6.15 | 15.99 |  -0.49 | {'lr': 1.0, 'depth': 20} |
| EmpkinS->Guardian | fus_ada+rule    | 6215 | 13.09 |        13.88 |            6.18 | 16.34 |  -1.21 | {'lr': 1.0, 'depth': 20} |
| EmpkinS->Guardian | sheikh_ada      | 6215 | 13.27 |        14.04 |            6.31 | 16.32 |  -0.99 | {'lr': 1.0, 'depth': 20} |
| EmpkinS->Guardian | sheikh_ada+rule | 6215 | 13.41 |        14.22 |            6.41 | 16.73 |  -1.66 | {'lr': 1.0, 'depth': 20} |
| EmpkinS->Guardian | fus_ridge       | 6215 | 15.96 |        16.61 |            5.13 | 15.93 |   0.16 | {'alpha': 10}            |
| EmpkinS->Guardian | fus_rf          | 6215 | 16.14 |        17.01 |            7.98 | 17.55 |  -5.94 | {'mf': 0.3, 'leaf': 5}   |
| EmpkinS->Guardian | sheikh_lgbm     | 6215 | 16.28 |        16.98 |            5.68 | 15.04 |  -0.7  | {'nl': 31, 'lr': 0.03}   |
| EmpkinS->Guardian | fus_lgbm        | 6215 | 16.37 |        17.16 |            5.92 | 15.15 |  -1.51 | {'nl': 31, 'lr': 0.03}   |
| EmpkinS->Guardian | sheikh_rf       | 6215 | 16.5  |        17.34 |            8.01 | 17.64 |  -6.12 | {'mf': 0.3, 'leaf': 5}   |
| EmpkinS->Guardian | fus_svr         | 6215 | 16.62 |        17.21 |            4.72 | 16.7  |   1.36 | {'C': 100}               |
| EmpkinS->Guardian | sheikh_svr      | 6215 | 17.43 |        18.06 |            5.3  | 17.92 |  -0.11 | {'C': 100}               |
| EmpkinS->Guardian | sheikh_ridge    | 6215 | 17.79 |        18.39 |            5.22 | 16.93 |  -2.83 | {'alpha': 10}            |
| EmpkinS->Guardian | abel_rf         | 6215 | 23.6  |        25.82 |           15.03 | 27.2  | -13.27 | {'mf': 1.0, 'leaf': 5}   |
| EmpkinS->Guardian | abel_svr        | 6215 | 28.42 |        31.41 |           24.05 | 39.93 | -16.03 | {'C': 100}               |
| Guardian->EmpkinS | fus_ada+rule    | 4980 | 20.03 |        19.51 |            5.68 | 16.74 |  13.09 | {'lr': 1.0, 'depth': 20} |
| Guardian->EmpkinS | fus_ada         | 4980 | 20.98 |        20.24 |            6.68 | 17.41 |  14.93 | {'lr': 1.0, 'depth': 20} |
| Guardian->EmpkinS | fus_svr         | 4980 | 23.44 |        22.22 |            8.15 | 19.05 |  13.05 | {'C': 100}               |
| Guardian->EmpkinS | sheikh_svr      | 4980 | 24.3  |        23.11 |            8.02 | 19.94 |  10.12 | {'C': 100}               |
| Guardian->EmpkinS | sheikh_ada+rule | 4980 | 26.28 |        25.91 |            5.91 | 20.59 |  20.09 | {'lr': 0.1, 'depth': 20} |
| Guardian->EmpkinS | sheikh_ada      | 4980 | 27.75 |        27.11 |            6.93 | 21.3  |  22.45 | {'lr': 0.1, 'depth': 20} |
| Guardian->EmpkinS | fus_lgbm        | 4980 | 28.98 |        27.88 |            8.62 | 18.1  |  22.5  | {'nl': 31, 'lr': 0.03}   |
| Guardian->EmpkinS | sheikh_lgbm     | 4980 | 29.16 |        28.07 |            8.75 | 18.65 |  22.09 | {'nl': 31, 'lr': 0.03}   |
| Guardian->EmpkinS | fus_ridge       | 4980 | 31.84 |        30.43 |            9.6  | 22.79 |  20.57 | {'alpha': 100}           |
| Guardian->EmpkinS | fus_rf          | 4980 | 33.94 |        32.4  |           12.47 | 21.57 |  28.89 | {'mf': 0.3, 'leaf': 5}   |
| Guardian->EmpkinS | sheikh_rf       | 4980 | 37.34 |        35.58 |           13.88 | 23.36 |  32.26 | {'mf': 0.3, 'leaf': 5}   |
| Guardian->EmpkinS | sheikh_ridge    | 4980 | 37.7  |        36    |           13.1  | 27.66 |  24.32 | {'alpha': 1000}          |
| Guardian->EmpkinS | abel_rf         | 4980 | 49.74 |        44.59 |           29.64 | 40.56 |  44.74 | {'mf': 1.0, 'leaf': 1}   |
| Guardian->EmpkinS | abel_svr        | 4980 | 66.5  |        58.59 |           45.78 | 58.59 |  63.69 | {'C': 100}               |

Abel 2026 reported (p.7): Emp->Gdn RF 36.24, SVR 19.39; Gdn->Emp RF 52.92 +/- 39.35, SVR 12.70 +/- 6.33 (read from the text; direction order of the first two as in the text).
