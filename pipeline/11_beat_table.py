"""Step 11 (advisor task 1): for EVERY patient -> every heartbeat with its R-peak, PEP and ICG waveform.
Outputs (all new files, inside dataset_beats/ and figures/per_patient/):
  dataset_beats/beats_all_patients.csv          one row per beat: patient, phase, beat id, R-peak sample + time, RR, HR, Q/B (both raters), PEP (both raters), QC flags
  dataset_beats/per_patient/<dataset>_<subject>.csv   same rows split by patient
  dataset_beats/waveforms/<dataset>_<subject>.npz     ICG waveform of every beat (R-100 ms .. R+600 ms, 1 kHz grid, filtered 0.5-40 Hz) + beat ids
  dataset_beats/patient_summary.csv             one row per patient
  figures/per_patient/<dataset>_<subject>.png   waveform figure per patient
PEP definition (as in Abel 2026): PEP = B-point - Q-wave onset, both from the manual annotation (rater 1; rater 2 given too).
RB = B-point time after R-peak.  Time axis of waveforms: ms relative to the R-peak."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

OUT = ROOT / "dataset_beats"; (OUT / "per_patient").mkdir(parents=True, exist_ok=True); (OUT / "waveforms").mkdir(exist_ok=True)
FIGP = FIG / "per_patient"; FIGP.mkdir(parents=True, exist_ok=True)
m, W, E = load_beats(cohort=False)
t = np.arange(W.shape[1]) - PRE                                            # ms relative to R

tab = pd.DataFrame({
    "dataset": m.dataset, "patient": m.subject, "condition": m.condition, "phase": m.phase, "beat_id": m.heartbeat_id,
    "fs_Hz": m.fs, "R_peak_sample": m.R, "R_peak_time_s": m.R / m.fs,
    "RR_prev_ms": m.RR_prev_ms, "HR_bpm": m.HR,
    "Q_onset_ms_after_R_rater1": m.Q1_ms, "B_point_ms_after_R_rater1": m.B1_ms, "RB_ms_rater1": m.B1_ms, "PEP_ms_rater1": m.PEP1,
    "B_point_ms_after_R_rater2": m.B2_ms, "PEP_ms_rater2": m.PEP2, "PEP_diff_raters_ms": (m.PEP1 - m.PEP2).abs(),
    "C_point_ms_after_R": m.C_ms, "C_amplitude": m.C_amp, "in_analysis_cohort": m.in_cohort,
})
tab["qc_note"] = np.where(m.B1_ms.isna(), "no B label", np.where(~m.in_cohort, "PEP outside 20-250 ms (label artefact)", "ok"))
tab.round(3).to_csv(OUT / "beats_all_patients.csv", index=False)

rows = []
phase_col = {}
for (ds, sub), g in tab.groupby(["dataset", "patient"], sort=True):
    idx = g.index.values; ok = g.in_analysis_cohort.values
    g.round(3).to_csv(OUT / "per_patient" / f"{ds}_{sub}.csv", index=False)
    np.savez_compressed(OUT / "waveforms" / f"{ds}_{sub}.npz", t_ms_after_R=t, beat_id=g.beat_id.values, phase=g.phase.values.astype(str),
                        icg=W[idx].astype(np.float32), PEP_ms_rater1=g.PEP_ms_rater1.values, in_analysis_cohort=ok)
    c = g[ok]
    rows.append(dict(dataset=ds, patient=sub, n_beats=len(g), n_usable=int(ok.sum()), n_phases=g.phase.nunique(),
                     HR_mean=c.HR_bpm.mean(), PEP_mean_ms=c.PEP_ms_rater1.mean(), PEP_sd_ms=c.PEP_ms_rater1.std(), PEP_min_ms=c.PEP_ms_rater1.min(), PEP_max_ms=c.PEP_ms_rater1.max(),
                     RB_mean_ms=c.RB_ms_rater1.mean(), RC_mean_ms=c.C_point_ms_after_R.mean(), inter_rater_PEP_MAE_ms=c.PEP_diff_raters_ms.mean()))
    # ---- per-patient figure ----
    Wc = W[idx][ok]; B = c.B_point_ms_after_R_rater1.values
    fig = plt.figure(figsize=(13, 4.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.3, 0.8])
    a = fig.add_subplot(gs[0]); sel = np.random.default_rng(0).choice(len(Wc), min(len(Wc), 150), replace=False)
    for i in sel: a.plot(t, Wc[i], color="#4C72B0", lw=0.5, alpha=0.18)
    a.plot(t, Wc.mean(0), color="k", lw=2, label=f"ensemble average (n={len(Wc)})")
    bm = np.nanmean(B); a.axvline(0, color="grey", lw=0.8, ls=":"); a.axvline(bm, color="crimson", lw=1.5, ls="--", label=f"mean B-point = R+{bm:.0f} ms")
    a.set_xlim(-50, 450); a.set_xlabel("time after R-peak (ms)"); a.set_ylabel("ICG  dZ/dt (filtered)"); a.set_title(f"{ds}  {sub}: ICG beats, R-aligned"); a.legend(frameon=False, fontsize=8)
    b = fig.add_subplot(gs[1]); phases = list(dict.fromkeys(c.phase)); cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, ph in enumerate(phases):
        cc = c[c.phase == ph]; b.scatter(cc.R_peak_time_s - c.R_peak_time_s.min(), cc.PEP_ms_rater1, s=6, color=cols[k % 10], label=ph)
    b.set_xlabel("time in recording (s)"); b.set_ylabel("PEP = B - Q (ms)"); b.set_title("PEP of every beat"); b.legend(frameon=False, fontsize=7, ncol=2, markerscale=2)
    h = fig.add_subplot(gs[2]); h.hist(c.PEP_ms_rater1, bins=30, color="#55A868", edgecolor="white"); h.set_xlabel("PEP (ms)"); h.set_title(f"mean {c.PEP_ms_rater1.mean():.0f} ± {c.PEP_ms_rater1.std():.0f} ms")
    for ax in fig.axes: ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGP / f"{ds}_{sub}.png", dpi=150); plt.close(fig)

S = pd.DataFrame(rows); S.round(2).to_csv(OUT / "patient_summary.csv", index=False)
print(f"{len(tab)} beats, {S.shape[0]} patients ({(S.dataset=='EmpkinS').sum()} EmpkinS, {(S.dataset=='Guardian').sum()} Guardian), usable beats {int(tab.in_analysis_cohort.sum())}")
print(S.round(1).to_string(index=False))
