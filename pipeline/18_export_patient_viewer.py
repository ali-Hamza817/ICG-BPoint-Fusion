"""Step 18: export Task-1 data (every beat of every patient: R-peak, PEP, waveform) as compact JSON for the shareable viewer (deploy_task1/).
Inputs: dataset_beats/beats_all_patients.csv, dataset_beats/waveforms/*.npz.  Waveform export: -50..+450 ms after R, every 5 ms (101 points), integer = value*1000."""
import json
import numpy as np, pandas as pd
from common import ROOT
D = ROOT / "dataset_beats"; OUT = ROOT / "deploy_task1" / "data"; OUT.mkdir(parents=True, exist_ok=True)
tab = pd.read_csv(D / "beats_all_patients.csv"); summ = pd.read_csv(D / "patient_summary.csv")
SL = slice(50, 551, 5)                                        # npz window is -100..+600 ms (1 ms steps): index 50 = -50 ms, 550 = +450 ms
rnd = lambda x, n=1: None if pd.isna(x) else round(float(x), n)
overview = []
for _, s in summ.iterrows():
    pid = f"{s.dataset}_{s.patient}"; g = tab[(tab.dataset == s.dataset) & (tab.patient == s.patient)].reset_index(drop=True)
    z = np.load(D / "waveforms" / f"{pid}.npz", allow_pickle=True); W = z["icg"][:, SL]; assert len(W) == len(g)
    z2 = np.load(ROOT / "dataset_task2" / "signals" / f"{pid}.npz", allow_pickle=True); EC = z2["ecg"][:, SL]; assert (z2["beat_id"] == g.beat_id.values).all()   # ECG windows come from the task-2 dataset
    ok = g.in_analysis_cohort.values.astype(bool); t0 = g.R_peak_time_s.min()
    phases = list(dict.fromkeys(g.phase))
    d = dict(id=pid, dataset=s.dataset, patient=s.patient, fs=int(g.fs_Hz.iloc[0]), phases=phases, t_ms=list(range(-50, 451, 5)),
             cols=["beat_id", "phase", "t_s", "RR_ms", "HR_bpm", "Q_ms", "B1_ms", "B2_ms", "PEP1_ms", "PEP2_ms", "C_ms", "ok"],
             beats=[[int(r.beat_id), phases.index(r.phase), rnd(r.R_peak_time_s, 2), rnd(r.RR_prev_ms, 0), rnd(r.HR_bpm, 1), rnd(r.Q_onset_ms_after_R_rater1, 0), rnd(r.B_point_ms_after_R_rater1, 0),
                     rnd(r.B_point_ms_after_R_rater2, 0), rnd(r.PEP_ms_rater1, 0), rnd(r.PEP_ms_rater2, 0), rnd(r.C_point_ms_after_R, 0), int(r.in_analysis_cohort)] for r in g.itertuples()],
             wf=np.round(W * 1000).astype(int).tolist(), ecg=np.round(EC * 1000).astype(int).tolist())
    json.dump(d, open(OUT / f"{pid}.json", "w"), separators=(",", ":"))
    mean = np.round(W[ok].mean(0) * 1000).astype(int).tolist()
    overview.append(dict(id=pid, dataset=s.dataset, patient=s.patient, n_beats=int(s.n_beats), n_usable=int(s.n_usable), phases=int(s.n_phases), HR=rnd(s.HR_mean, 1), PEP=rnd(s.PEP_mean_ms, 1), PEP_sd=rnd(s.PEP_sd_ms, 1),
                         PEP_min=rnd(s.PEP_min_ms, 0), PEP_max=rnd(s.PEP_max_ms, 0), RB=rnd(s.RB_mean_ms, 1), RC=rnd(s.RC_mean_ms, 1), IR_MAE=rnd(s.inter_rater_PEP_MAE_ms, 1), mean=mean))
json.dump(dict(patients=overview, t_ms=list(range(-50, 451, 5))), open(OUT / "patients.json", "w"), separators=(",", ":"))
print(len(overview), "patients;", sum(f.stat().st_size for f in OUT.glob("*.json")) / 1e6, "MB total")
