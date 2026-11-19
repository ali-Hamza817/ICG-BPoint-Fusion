# Sheikh's method on Abel's public data: summary for discussion

All numbers come from saved predictions (`results/tasks_v3/`, `results/ea_vs_single_w*.csv`), produced by the scripts in `pipeline/`. Full tables: `reports/V3_RESULTS.md`.

## What was built (advisor's directions)
| Direction | Status |
|---|---|
| Per-patient beats: R-peak, PEP, waveform for every beat of every patient | Done: `dataset_beats/` (12,014 beats, 39 patients), `figures/per_patient/` |
| Feature lists, Abel vs Sheikh | Done: `reports/FEATURE_COMPARISON.md` |
| Abel's real algorithms | Done: his 12 files run unmodified via a small shim (`external/biopsykit_shim`) |
| Sheikh's real feature code | Done: line-by-line port of his MATLAB (`03c`), exact match (0.0 deviation) on his 2 demo beats |
| Sheikh-style AdaBoost (StandardScaler, his hyper-parameter region, featureless-beat rule) | Done |
| Ensemble-averaged beats with his noise-removal algorithm | Done (10 s / 20 s / 30 s windows) |

## Result 1: single-beat B-point error, subject-wise 5-fold CV, 11,195 beats (rater-1 B-point)
| Model | MAE (ms) |
|---|---|
| Sheikh 465 features + AdaBoost | 7.02 |
| Fusion (465 + Abel 13) + AdaBoost | 7.04 |
| Fusion + RF | 8.11 |
| Sheikh 465 + RF | 8.16 |
| Sheikh AdaBoost + his featureless-beat rule | 8.43 |
| **Abel replica (his real 12 rules + RR, RF)** | **10.17** |
| Abel rules + SVR | 11.88 |

Paired per-subject comparison, Sheikh AdaBoost vs Abel replica: mean difference -3.19 ms, 95 % CI [-3.83, -2.53], Wilcoxon p < 0.001, 36 of 39 subjects improve.

## Result 2: cross-dataset (train on one dataset only, test on the other; nothing selected on the target)
| | EmpkinS -> Guardian | Guardian -> EmpkinS |
|---|---|---|
| Fusion AdaBoost | 13.00 | 20.98 (20.03 with featureless rule) |
| Sheikh AdaBoost | 13.27 | 27.75 |
| Abel replica RF | 23.60 | 49.74 |
| Abel replica SVR | 28.42 | 66.50 |
| *Abel published RF / SVR* | *36.24 / 19.39* | *52.92 / 12.70* |

Guardian -> EmpkinS shows a large systematic bias for every model (+13 to +22 ms for the AdaBoost variants).

## Result 3: single beat vs ensemble-averaged beat (reference = mean rater B-point of the surviving beats in a window)
| Window | EA model (train+test on averaged beats), best / AdaBoost | Single-beat AdaBoost, predictions averaged over the window | Same model, mean per-beat error |
|---|---|---|---|
| 10 s (469 windows) | 7.97 (SVR) / 9.40 | 4.70 | 6.98 |
| 20 s (258 windows) | 8.45 (SVR) / 9.28 | 5.07 | 7.86 |

## Caveats (must be stated with any claim)
1. **My Abel replica (10.17 ms) does not reproduce Abel's published 8.13 ms.** Cause unknown (heartbeat segmentation, cohort, folds). The paired comparison is fair because both run on identical beats and folds, but my replica number is not his number.
2. Abel's C-point extractor (fixed prominence 1) fails on 10.1 % of Guardian beats (0.2 % EmpkinS); beats where all 12 rules fail are excluded as Abel does. Evaluation set 11,195 beats (Abel: 11,239).
3. Heartbeats = annotated reference heartbeats with our R-peaks, not PEPbench's neurokit segmentation + matching.
4. AdaBoost: one repeat, small nested grid inside Sheikh's published range (max_features 0.7, depth 12/20, learning rate 0.1/1; 100 trees inner, 500 final). Fast models: three repeats.
5. Annotated beats span only ~30 s per phase, so Sheikh's 60 s windows are impossible; 10/20/30 s used. The averaged-beat label is our assumption (mean of rater B-points), not an annotation of the averaged beat. The averaged-beat models train on 258 to 469 samples, versus ~9,000 single beats, so Result 3 does not show that averaging is harmful; it shows averaged-beat training is data-starved here.
6. Sheikh's amplitude thresholds in the averaging algorithm remove 41 % of EmpkinS beats at stage 1 (Guardian 6 %), a signal-scale mismatch.
7. Abel's published cross-dataset SVR for Guardian -> EmpkinS (12.70 ms) is better than anything here (20.03 ms); my SVR replica (66.50 ms) does not reproduce his SVR.
8. Sheikh's own datasets are not available; his published 3.5 ms (averaged beats, Emory) is not comparable with any number above.
