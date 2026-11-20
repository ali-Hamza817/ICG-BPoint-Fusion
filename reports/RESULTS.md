# Results: our models vs Abel et al. 2026 (published figures)

All numbers are computed from saved predictions on the **clean beats** (artefact-labelled points removed, see README). Abel's figures are taken from his paper as published (MAE against the manual B-point annotation); his cohort (11,239 cycles) and folds differ from ours (11,694 usable beats, 11,018 evaluated here), so differences are **indicative, not like-for-like**.

## 1. Pooled single-beat B-point error (subject-wise cross-validation)

| Model | MAE ms | SD | Median AE | per-subject MAE [95 % CI] | vs Abel published 8.13 |
|---|---|---|---|---|---|
| Sheikh 465 + Abel 13 features + AdaBoost | **6.73** | 12.78 | 2.00 | 7.26 [5.85, 8.92] | -1.40 ms |
| Sheikh 465 features + AdaBoost | **6.73** | 12.69 | 2.00 | 7.24 [5.82, 8.97] | -1.40 ms |
| Fusion + Random Forest | **7.74** | 11.69 | 2.93 | 8.21 [6.94, 9.80] | -0.39 ms |
| Sheikh 465 + Random Forest | **7.82** | 11.88 | 2.95 | 8.29 [6.92, 9.89] | -0.31 ms |
| Fusion + LightGBM | **8.21** | 11.52 | 3.63 | 8.56 [7.19, 10.04] | +0.08 ms |
| Sheikh 465 + LightGBM | **8.28** | 11.66 | 3.55 | 8.60 [7.22, 10.21] | +0.15 ms |
| Fusion + SVR | **10.16** | 12.41 | 5.17 | 10.70 [9.28, 12.35] | +2.03 ms |
| Sheikh 465 + SVR | **10.63** | 13.15 | 5.24 | 11.26 [9.70, 13.07] | +2.50 ms |

Published (Abel 2026, Random Forest on 12 rule-based estimates + RR): **MAE 8.13 ± 12.12 ms** (CV-fold mean 8.15). Published inter-rater MAE 8.35 ± 14.78 ms; on our clean beats the inter-rater B-point difference is 7.94 ± 14.26 ms.

Uncertainty: the sample-wise MAE above is pooled over beats; the per-subject mean with its bootstrap interval over the 39 subjects is wider. For our best model the interval [5.85, 8.92] ms **contains** Abel's published 8.13 ms, so with 39 subjects the difference to the published figure is **not** clearly outside sampling uncertainty (and the cohorts differ).

AdaBoost: one repeat, nested tuning on a small grid inside Sheikh's published range (max_features 0.7, depth 12/20, learning rate 0.1/1; 100 trees in the inner loop, 500 for the final model). Other models: mean of three repeats.

## 2. Cross-dataset (train on one dataset only, test on the other)

| Direction | Our AdaBoost (nested pick) | Our AdaBoost, all 4 settings (min – max) | Abel published RF | Abel published SVR |
|---|---|---|---|---|
| EmpkinS → Guardian · Sheikh 465 | 13.05 | 13.05 – 14.08 | 36.24 | 19.39 |
| EmpkinS → Guardian · Fusion | 12.46 | 12.46 – 13.88 | 36.24 | 19.39 |
| Guardian → EmpkinS · Sheikh 465 | 31.59 | 29.22 – 34.49 | 52.92 | 12.7 |
| Guardian → EmpkinS · Fusion | 28.22 | 25.40 – 33.12 | 52.92 | 12.7 |

Reading: EmpkinS → Guardian is clearly below Abel's published values. Guardian → EmpkinS is **not** better than Abel's published SVR (12.70 ms); the result depends strongly on the model setting chosen, which is why the full range is shown. Every model shows a systematic bias in that direction (the two datasets differ strongly in B-point timing).

## 3. Single beat vs ensemble-averaged beat (Sheikh's averaging algorithm on Abel's beats)

Reference = mean rater-1 B point of the beats in each window (our assumption, raters annotated single beats).

| Window | Model trained on averaged beats (best / AdaBoost) | Single-beat AdaBoost, predictions averaged over the window | Same model, mean per-beat error |
|---|---|---|---|
| 10 s (468 windows) | 7.92 (sheikh_svr) / 9.31 | 4.71 | 6.88 |
| 20 s (258 windows) | 8.46 (sheikh_svr) / 9.34 | 4.97 | 7.76 |

Averaged-beat models train on only a few hundred averaged beats (versus about 9,000 single beats), so this does not show that averaging is harmful; it shows that averaged-beat training is data-starved here.

## Caveats
- Published figures are used as reference only; no re-implementation of Abel's pipeline is reported.
- Different cohort size and folds from Abel's; single AdaBoost repeat; cross-dataset results depend on the model setting.
- Of 12,014 annotated beats, 209 were removed from the usable set because a rater's B or Q point was an 'Artefact' marker.
