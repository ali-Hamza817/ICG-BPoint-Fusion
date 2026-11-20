"""Step 19 (advisor task 2): per-beat dataset  ECG signal + ICG signal + B (rater 1) + PEP (rater 1) + C value  for every patient,
exported (a) for Python (CSV + per-patient NPZ), (b) in the EXACT input format of Sheikh's feat_vec_465.m (MATLAB .mat, one struct 'Rec_feat' per beat), and
(c) the 465-feature matrix computed from these inputs with the Python port of feat_vec_465.m (pipeline/03c, validated bit-exact on his 2 demo beats).
Signals: 1 kHz grid, window R-100 ms .. R+600 ms (index j <-> time j-100 ms after the R-peak).  ICG = dZ/dt band-pass 0.5-40 Hz (Sheikh's filter), ECG = neurokit-cleaned.
Definitions (as Abel 2026 / Sheikh 2022): RB = B-point time after R;  PEP = B - Q-onset (manual annotation, rater 1);  C = main ICG peak: RC = time after R, C amplitude = ICG value there;
vel_index = C amplitude - ICG amplitude at B  (this is what feat_vec_465.m expects: mag_C = vel_index + feat_icg(RB_out+1)).
His MATLAB code indexes feat_icg(RB_out+1) in a 151-sample vector, so beats with RB_out > 150 ms or < 0 cannot be processed by that file ('matlab_ok' = 0); the Python port handles all beats.
Outputs: dataset_task2/ (task2_beats.csv, signals/*.npz, matlab_demo_format_<dataset>.zip, features_465.npy + features_465_index.csv + features_465_columns.txt, README.md)"""
import importlib.util, zipfile, io
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from common import *

spec = importlib.util.spec_from_file_location("f3c", Path(__file__).with_name("03c_sheikh_features_v2.py")); F3C = importlib.util.module_from_spec(spec); spec.loader.exec_module(F3C)
OUT = ROOT / "dataset_task2"; (OUT / "signals").mkdir(parents=True, exist_ok=True)
m, W, E = load_beats(cohort=False)                                   # all 12,014 annotated beats (flag in_cohort marks the 11,903 usable ones)
T = np.arange(-PRE, POST + 1)
RB = m.B1_ms.values; RBi = np.where(np.isnan(RB), -999, np.round(RB)).astype(int)
inwin = (RBi >= -PRE) & (RBi <= POST)
b_amp = np.full(len(m), np.nan); b_amp[inwin] = W[np.where(inwin)[0], PRE + RBi[inwin]]
vel = m.C_amp.values - b_amp
matlab_ok = ((RBi >= 0) & (RBi <= 150) & m.in_cohort.values)

tab = pd.DataFrame({"row": m.row, "dataset": m.dataset, "patient": m.subject, "condition": m.condition, "phase": m.phase, "beat_id": m.heartbeat_id, "fs_Hz": m.fs.astype(int),
                    "R_sample": m.R, "R_time_s": (m.R / m.fs).round(3), "RR_prev_ms": m.RR_prev_ms, "HR_bpm": m.HR.round(1),
                    "Q_ms_after_R_rater1": m.Q1_ms, "B_ms_after_R_rater1": m.B1_ms, "PEP_ms_rater1": m.PEP1,
                    "B_ms_after_R_rater2": m.B2_ms, "PEP_ms_rater2": m.PEP2,
                    "C_ms_after_R": m.C_ms, "C_amplitude": m.C_amp.round(5), "ICG_at_B_amplitude": np.round(b_amp, 5), "vel_index_C_minus_B": np.round(vel, 5),
                    "usable": m.in_cohort.astype(int), "matlab_ok": matlab_ok.astype(int)})
tab.to_csv(OUT / "task2_beats.csv", index=False)

# ---- per-patient NPZ (ECG + ICG windows + labels) ----
for (ds, sub), g in tab.groupby(["dataset", "patient"]):
    ix = g.index.values
    np.savez_compressed(OUT / "signals" / f"{ds}_{sub}.npz", t_ms_after_R=T, ecg=E[ix].astype(np.float32), icg=W[ix].astype(np.float32), beat_id=g.beat_id.values, phase=g.phase.values.astype(str),
                        R_sample=g.R_sample.values, fs_Hz=g.fs_Hz.values, Q_ms_rater1=g.Q_ms_after_R_rater1.values, B_ms_rater1=g.B_ms_after_R_rater1.values, PEP_ms_rater1=g.PEP_ms_rater1.values,
                        B_ms_rater2=g.B_ms_after_R_rater2.values, PEP_ms_rater2=g.PEP_ms_rater2.values, C_ms=g.C_ms_after_R.values, C_amplitude=g.C_amplitude.values, usable=g.usable.values)

# ---- MATLAB demo format (one struct 'Rec_feat' per beat, exactly the fields used by feat_vec_465.m) ----
def rec(i):
    s = W[i].astype(float); d1 = np.gradient(s); d2 = np.gradient(d1); sl = slice(PRE, PRE + 151)
    pid = f"{m.subject.iloc[i]}_{int(m.heartbeat_id.iloc[i]):04d}"
    return dict(id=pid, feat_icg=s[sl], feat_icg_D1=d1[sl], feat_icg_D2=d2[sl], feat_icg_D2_sign=np.sign(d2[sl]), vel_index=float(vel[i]), RC=float(m.C_ms.iloc[i]), RB_out=float(RBi[i])), (s, d1, d2)
rows_ok = np.where(matlab_ok)[0]
for ds in ("EmpkinS", "Guardian"):
    with zipfile.ZipFile(OUT / f"matlab_demo_format_{ds}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for i in rows_ok[m.dataset.values[rows_ok] == ds]:
            r, _ = rec(i); bio = io.BytesIO(); sio.savemat(bio, {"Rec_feat": r}); z.writestr(f"{ds}_{m.subject.iloc[i]}/{r['id']}.mat", bio.getvalue())

# ---- 465 features via the Python port of feat_vec_465.m, fed with exactly the exported inputs ----
use = np.where(m.in_cohort.values)[0]; X = np.zeros((len(use), 465), np.float32); aux = []; dev = []
for k, i in enumerate(use):
    s = W[i].astype(float); d1 = np.gradient(s); d2 = np.gradient(d1); sl = slice(PRE, PRE + 151)
    ic = s[sl]; RCi = float(m.C_ms.iloc[i]); magC = float(m.C_amp.iloc[i])                    # window route: C amplitude directly
    v, a = F3C.sheikh_core(ic, d1[sl], d2[sl], np.sign(d2[sl]), RCi, magC); X[k] = v; aux.append(a)
    if matlab_ok[i]:                                                                          # his-format route: mag_C = vel_index + feat_icg(RB_out+1)
        magC2 = vel[i] + ic[RBi[i]]; v2, _ = F3C.sheikh_core(ic, d1[sl], d2[sl], np.sign(d2[sl]), RCi, float(magC2)); dev.append(np.abs(v - v2).max())
np.save(OUT / "features_465.npy", X)
idx = tab.iloc[use][["row", "dataset", "patient", "phase", "beat_id", "B_ms_after_R_rater1", "PEP_ms_rater1", "C_ms_after_R", "matlab_ok"]].copy()
idx["has_inflection"] = [a["has_infl"] for a in aux]; idx["has_notch"] = [a["has_notch"] for a in aux]; idx.to_csv(OUT / "features_465_index.csv", index=False)
open(OUT / "features_465_columns.txt", "w").write("\n".join(F3C.NAMES) + "\n")
print(f"beats {len(tab)} (usable {int(tab.usable.sum())}), patients {tab.patient.nunique()}; MATLAB-compatible {int(matlab_ok.sum())}; features {X.shape}")
print("his-format route vs window route: max abs deviation over %d beats = %.3g" % (len(dev), max(dev)))
print("flagged for his MATLAB file: RB>150 or <0 among usable:", int(((RBi > 150) | (RBi < 0))[m.in_cohort.values].sum()))
