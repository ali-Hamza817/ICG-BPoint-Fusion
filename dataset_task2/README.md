# Task 2 dataset: ECG + ICG + B (rater 1) + PEP (rater 1) + C for every beat of every patient

Source: public datasets of Abel et al. 2026 (EmpkinS n=15, Guardian n=24). 12,014 annotated heartbeats, 11,694 usable (clean B and Q label, not an artefact marker, PEP between 20 and 250 ms), 39 patients.

## Definitions
- Time axis of every waveform: ms relative to the R-peak (window R-100 ms .. R+600 ms, 1 kHz grid, index j = time j-100 ms).
- ICG = dZ/dt, 4th-order Butterworth band-pass 0.5-40 Hz (Sheikh's filter). ECG = neurokit-cleaned. Guardian (500 Hz) is spline-upsampled to the 1 kHz grid.
- B (RB) = B-point time after R, rater 1. Q = Q-wave onset after R, rater 1. **PEP = B - Q** (rater 1). Rater 2 columns are given for comparison.
- C = main ICG peak: RC = time after R, C amplitude = ICG value there. `vel_index_C_minus_B` = C amplitude - ICG amplitude at B (the input feat_vec_465.m expects).

## Files
| File | Content |
|---|---|
| `task2_beats.csv` | one row per beat: patient, phase, beat id, R sample/time, RR, HR, Q, B, PEP (rater 1 and 2), C time + amplitude, vel_index, `usable`, `matlab_ok` |
| `signals/<dataset>_<patient>.npz` | per patient: `ecg` and `icg` (beats x 701), `t_ms_after_R`, and the labels above as arrays |
| `matlab_demo_format_<dataset>.zip` | the SAME beats in the exact input format of `feat_vec_465.m` (one MATLAB struct `Rec_feat` per beat with `id, feat_icg, feat_icg_D1, feat_icg_D2, feat_icg_D2_sign, vel_index, RC, RB_out`, identical field layout to his `Demo_102D3_B_*.mat`) |
| `features_465.npy` + `features_465_index.csv` + `features_465_columns.txt` | the 465-feature vector of every usable beat, computed with a line-by-line Python port of `feat_vec_465.m` (11,694 x 465, float32, column names in the txt; index csv gives patient/beat/label for each row) |

## Using his MATLAB code directly
Unzip a `matlab_demo_format_*.zip`, point `demo_beat_folder` in `feat_vec_465.m` to a patient folder (the script processes every .mat in the folder, as for his demo files).
**His file cannot process beats with RB_out > 150 ms or < 0** (it reads `feat_icg(RB_out+1)` from a 151-sample vector). Those 261 usable beats (2.2 %, mostly Guardian) are therefore not in the zips (`matlab_ok = 0`); the Python features cover all 11,694 beats.

## Validation of the Python port (MATLAB/Octave are not installed on the analysis server)
- `pipeline/12_validate_sheikh_port.py`: reproduces his own MATLAB output (`Demo Data/features_final_demo_465.xlsx`) with 0.0 deviation on his two demo beats (this does not exercise the 'inflection found' branch).
- Feeding the exported demo-format inputs through the port gives exactly the same 465 features as computing them from the waveform windows (max abs deviation 0 over 11,433 beats).
- Running his actual `.m` file on these `.mat` files has NOT been done here; a comparison on a few beats in MATLAB would be the final proof.

## Code
`pipeline/19_task2_dataset.py` (dataset + features), `pipeline/20_task2_plots.py` (figures in `figures/task2/`), `pipeline/03c_sheikh_features_v2.py` (port of `feat_vec_465.m`).
