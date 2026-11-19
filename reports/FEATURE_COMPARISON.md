# Features: Abel et al. 2026 vs Sheikh et al. 2022

Sources (verified in the PDFs in this repo): Abel 2026 Sec. 2.2 (p.3), Table 1 (p.4), Table 3 SHAP (p.8). Sheikh 2022 Sec. 2.3.1 (p.4), search window (p.5), Table B1 (p.21, Appendix B).

## 1. At a glance

| | **Sheikh 2022** | **Abel 2026** |
|---|---|---|
| Input to the model | **Ensemble-averaged** ICG beat (3-stage EA algorithm) | **Single** ICG beat (beat-to-beat) |
| Number of features | **465** | **13** (12 rule-based B-point estimates + RR interval) |
| What a feature is | A *measurement on the waveform* (samples, slopes, shape landmarks) | The *answer of another B-point algorithm* (a guess in ms) |
| Waveform samples used directly | **Yes**: ICG, 1st and 2nd derivative over [R, R+150 ms] = 450 values | **No** |
| Morphology features | **Yes**: last notch, last inflection, max of 3rd derivative (rapid slope change), C amplitude, RC interval + amplitude/time differences (15) | **No** (only implicitly, inside the rules) |
| Search window | **Variable**: [R+35 ms, min(R+140 ms, t(0.66 x C amplitude))] | Fixed per rule (e.g. C-150 ms ... C) |
| Model | AdaBoost regressor (best), also RF, SVR, Ridge, LASSO, kernel ridge | Random Forest (best), also k-NN, SVR, DT |
| Target | B-point (PEP) of the averaged beat | B-point of the single beat |
| Validation | Subject-wise nested CV, 10 outer / 5 inner folds | Subject-wise nested CV, 5 outer / inner; cross-dataset test both directions |
| Result | MAE 3.5 +/- 9.2 ms (averaged beats, Emory, private) | MAE 8.13 +/- 12.12 ms (single beats, public) |

Key difference in one sentence: **Sheikh describes the waveform and lets the model find the B-point; Abel lets 12 existing algorithms find the B-point and lets the model decide whom to trust.**

## 2. Sheikh's 465 features (Table B1)

| Index | Group | Feature |
|---|---|---|
| 1-151 | Time series | ICG dZ/dt samples in [R, R+150 ms] (1 sample / ms) |
| 152-301 | Time series | 1st derivative d2Z/dt2 in [R, R+150 ms] |
| 302-450 | Time series | 2nd derivative d3Z/dt3 in [R, R+150 ms] |
| 451 | High level, amplitude in search window | Last notch |
| 452 | | Last inflection point |
| 453 | | Maximum of d3Z/dt3 (rapid slope change) |
| 454 | | C-point amplitude (max ejection velocity) |
| 455-457 | Amplitude differences | C minus last notch / last inflection / max of d3Z/dt3 |
| 458 | Time intervals | t(last notch) - t(last inflection) |
| 459-461 | | tRC - t(last notch) / t(last inflection) / t(max d3Z/dt3) |
| 462 | | RC interval (initial systolic time interval) |
| 463-465 | Time index in search window | Last notch / last inflection / max of d3Z/dt3 |

Conventions from the text: inflection = positive 1st derivative and 3rd derivative changes sign (-) to (+), last one in the window; notch = negative peak, last one; missing landmark -> feature set to 0.

## 3. Abel's 13 features (Table 1)

| Feature | Source | Idea |
|---|---|---|
| Arb17IC | Arbol 2017 | Last crossing of the cycle mean before C |
| Arb17SD | Arbol 2017 | Peak of 2nd derivative 150-100 ms before C |
| Arb17TD | Arbol 2017 | Peak of 3rd derivative in window starting 300 ms before C |
| Deb93 | Debski 1993 | Local minimum of 2nd derivative nearest C |
| Dro22 | Drost 2022 | Point of max distance to the line C-150 ms ... C (**SHAP importance 53.9 %**) |
| For18 | Forouzanfar 2018 | Local max / zero crossing in d3Z/dt3 in most prominent rising segment A..C |
| Loz07LR | Lozano 2007 | RB = 0.55 RC + 4.45 |
| Loz07QR | Lozano 2007 | RB = 1.233 RC - 0.0032 RC^2 - 31.59 |
| Mil22 | Miljkovic 2022 | Weighted-window approach |
| Pal21 | Pale 2021 | Search C-80 ms ... half-amplitude; closest local min / slope rule |
| She90 | Sherwood 1990 | Zero crossing nearest C |
| Ste85 | Stern 1985 | Local minimum of dZ/dt nearest C |
| RR interval | | R-R distance to the previous beat |

Missing values: a rule that fails gives NaN. Exp. 1 keeps NaN (RF), Exp. 2 imputes by the median of the other rules in the same beat. Beats where all rules fail are dropped (372, 3.2 %).

## 4. Do the two feature sets overlap?

- **Shared ideas:** the C-point and RC interval (Sheikh feature 462 vs Loz07 rules), the 2nd/3rd derivative landmarks (Sheikh 451-453, 463-465 vs Deb93, Arb17SD/TD, For18), notch/minimum nearest C (Sheikh 451, 463 vs Ste85).
- **Only Sheikh:** the 450 raw waveform/derivative samples, amplitude differences, and the variable search window.
- **Only Abel:** the RR interval, and the regression-formula, chord-distance (Dro22) and isoelectric-crossing rules.

## 5. Consequences for our experiment (the advisor's question T4)

1. Abel's model never sees the waveform itself; Sheikh's does. Applying Sheikh's features to Abel's data is therefore a different information source, not just a different model.
2. Sheikh's features were designed for averaged beats. On single beats, the landmark features (notch, inflection) are noisier, so they may help less than in the paper. This is directly testable (single beat vs ensemble-averaged beat, same data).
3. In our code (`pipeline/03_sheikh_features.py`) the 465-feature layout follows this table; the parts the paper leaves open are marked ASSUMPTION in the code.
4. Our re-implementation of Abel's rules (`pipeline/08_abel_features.py`): Table 3 of Abel confirms Deb93 = 2nd-derivative minimum, Arb17SD = 2nd-derivative peak, Arb17TD = 3rd-derivative peak (match our code). **For18 is defined on d3Z/dt3 in Table 3; our current implementation works on the signal, and should be corrected.** Mil22 is not implemented.
