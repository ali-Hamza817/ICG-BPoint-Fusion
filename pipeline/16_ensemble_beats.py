"""Step 16 (advisor direction 3): ensemble-averaged (EA) ICG beats on Abel's data with Sheikh's OWN three-stage algorithm
(port of external/sheikh_toolbox_B_point/ICG_Noise_Removal/Noise_removal.m, steps 6-8) + Sheikh's 465 features on the EA beat.
Per phase segment: ICG band-pass 0.5-40 Hz (his filter), 1 kHz grid; consecutive non-overlapping windows of WIN_S seconds (default 20 s; Sheikh: 60 s) (first window starts 150 ms before the first R);
beat = [R-150 ms, R + median(RR of window) - 150 ms);  stage 1 amplitude filter (max<0.4 | max>3 | min<-2 deleted), stage 2 coarse (|lag|>=50 ms or r<=0.5 deleted),
stage 3 fine (circshift by lag; delete if (|lag|<=1 & r<=0.8) | |lag|>1);  EA beat = mean of survivors.
LABEL of an EA beat (ASSUMPTION - raters annotated single beats, not EA beats): mean of the rater-1 B-point (ms after R) over surviving beats that carry a valid label.
DEVIATIONS: R-peaks = our R of the annotated beats (not jqrs); windows are phase-local; Guardian (500 Hz) is spline-upsampled to 1 kHz after filtering at 500 Hz.
Outputs: data/processed/ea_meta_w<S>.csv, ea_X_w<S>.npy (465 features), ea_beats_w<S>.pkl (EA waveforms); usage: python 16_ensemble_beats.py <window seconds>"""
import importlib.util, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from common import *


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
M13 = _load("m13", "13_abel_real_features.py"); F3C = _load("f3c", "03c_sheikh_features_v2.py")
FS1K, LAG_THRES = 1000, 50
WIN_S = int(sys.argv[1]) if len(sys.argv) > 1 else 20          # DEVIATION: Sheikh uses 60 s, but annotated beats per phase span only ~30 s (about 40 beats) -> shorter windows
TAG = f"_w{WIN_S}"


def xcorr_lag(x, y):
    c = correlate(x, y, mode="full"); lags = correlation_lags(len(x), len(y), mode="full"); return int(lags[int(np.argmax(c))])


def three_stage(B):
    """B: (n_beats, L) ICG beats. Returns EA beat (or None), indices of surviving beats, counts removed per stage. Port of Noise_removal.m."""
    n = len(B); idx = np.arange(n); cnt = [0, 0, 0]
    mx, mn = B.max(1), B.min(1)
    k1 = ~((mx < 0.4) | (mx > 3) | (mn < -2)); cnt[0] = int((~k1).sum()); B1, i1 = B[k1], idx[k1]
    if len(B1) == 0: return None, i1, cnt
    ea1 = B1[0] if len(B1) == 1 else B1.mean(0)
    keep2 = []
    for r in B1:
        lag = xcorr_lag(ea1, r); cc = np.corrcoef(ea1, r)[0, 1]
        keep2.append(not (abs(lag) >= LAG_THRES or cc <= 0.5))
    keep2 = np.array(keep2); cnt[1] = int((~keep2).sum()); B2, i2 = B1[keep2], i1[keep2]
    if len(B2) == 0: return None, i2, cnt
    ea2 = B2[0] if len(B2) == 1 else B2.mean(0)
    sh, keep3 = [], []
    for r in B2:
        lag = xcorr_lag(ea2, r); rs = np.roll(r, lag); lag_s = xcorr_lag(ea2, rs); cc = np.corrcoef(ea2, rs)[0, 1]
        sh.append(rs); keep3.append(not (((abs(lag_s) <= 1) and cc <= 0.80) or abs(lag_s) > 1))
    keep3 = np.array(keep3); cnt[2] = int((~keep3).sum()); B3, i3 = np.array(sh)[keep3], i2[keep3]
    if len(B3) == 0: return None, i3, cnt
    return (B3[0] if len(B3) == 1 else B3.mean(0)), i3, cnt


def seg_job(key, icg_raw, fs, g):
    """g: beats of this segment (rows of beats_meta) sorted by R."""
    f = icg_filter(icg_raw.astype(float), fs)
    if fs == 500: f = CubicSpline(np.arange(len(f)), f)(np.arange(0, 2 * (len(f) - 1) + 1) / 2.0)
    R = np.round(g.R.values * (FS1K / fs)).astype(int); out = []
    t0 = R[0] - 150
    for k in range(int((R[-1] - t0) // (WIN_S * FS1K))):
        lo, hi = t0 + k * WIN_S * FS1K, t0 + (k + 1) * WIN_S * FS1K
        sel = np.where((R >= lo) & (R < hi))[0]
        if len(sel) < 10: continue
        Rw = R[sel]; med = int(np.ceil(np.median(np.diff(Rw)))); ul = med - 150
        if ul < 300: continue
        rows = [i for i in range(len(Rw) - 1) if Rw[i] - 150 >= 0 and Rw[i] + ul - 1 < len(f)]      # last R skipped (as in MATLAB)
        if len(rows) < 8: continue
        B = np.array([f[Rw[i] - 150: Rw[i] + ul] for i in rows])
        ea, surv, cnt = three_stage(B)
        if ea is None: continue
        gs = g.iloc[sel[rows]]; gsurv = gs.iloc[surv]
        lab = gsurv[(gsurv.B1_ms.notna()) & ((gsurv.B1_ms - gsurv.Q1_ms).between(20, 250))]
        if len(lab) < 3: continue
        out.append(dict(dataset=key[0], subject=key[1], condition=key[2], phase=key[3], window=k, n_total=len(rows), n_s1=cnt[0], n_s2=cnt[1], n_s3=cnt[2], n_surv=len(surv),
                        BCF=len(surv) / len(rows), n_labelled=len(lab), RB_label=float(lab.B1_ms.mean()), RB_label_sd=float(lab.B1_ms.std()), PEP_label=float((lab.B1_ms - lab.Q1_ms).mean()),
                        HR=float(60000 / np.median(np.diff(Rw))), rr_med_ms=med, beat_rows=g.index[sel[rows]][surv].tolist(), ea=ea.astype(np.float32)))
    return out


def job(j, meta):
    res = []
    try:
        gen = M13.segments_guardian(j[1]) if j[0] == "g" else M13.segments_empkins(*j[1])
        for key, icg, fs in gen:
            g = meta[(meta.dataset == key[0]) & (meta.subject == key[1]) & (meta.condition.fillna("") == key[2]) & (meta.phase == key[3])].sort_values("R")
            if len(g) < 25: continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); res += seg_job(key, icg, fs, g)
    except Exception as e:
        print("JOB FAIL", j, repr(e)[:200], flush=True)
    return res


if __name__ == "__main__":
    meta = pd.read_csv(PROC / "beats_meta.csv")
    jobs = [("g", p.name) for p in sorted((GDN / "data_raw").iterdir()) if p.is_dir()]
    jobs += [("e", (v.name, c)) for v in sorted((EMP / "data_per_subject").iterdir()) if v.is_dir() for c in ["tsst", "ftsst"]]
    R = [o for r in Parallel(n_jobs=24, backend="loky")(delayed(job)(j, meta) for j in jobs) for o in r]
    X, keep = [], []
    for i, o in enumerate(R):
        ea = o["ea"].astype(float); w = ea[150 - PRE:]                                  # window with R at index PRE=100 (as in common.PRE)
        if len(w) < PRE + 400 + 1: keep.append(False); continue
        seg = w[PRE + 50: PRE + 401]; c = int(np.argmax(seg)) + 50
        v, aux = F3C.sheikh_from_window(w, PRE, float(c), float(w[PRE + c])); X.append(v); keep.append(True)
        o.update(C_ms=c, C_amp=float(w[PRE + c]), has_infl=aux["has_infl"], has_notch=aux["has_notch"], RB_max_2D=float(v[-1] - 1.0))
    R = [o for o, k in zip(R, keep) if k]; X = np.array(X)
    pickle.dump([o["ea"] for o in R], open(PROC / f"ea_beats{TAG}.pkl", "wb"))
    df = pd.DataFrame([{k: v for k, v in o.items() if k != "ea"} for o in R]); df["beat_rows"] = df.beat_rows.map(lambda x: ";".join(map(str, x)))
    df.to_csv(PROC / f"ea_meta{TAG}.csv", index=False); np.save(PROC / f"ea_X{TAG}.npy", X)
    print(f"EA beats: {len(df)} from {df.subject.nunique()} subjects ({(df.dataset=='EmpkinS').sum()} EmpkinS, {(df.dataset=='Guardian').sum()} Guardian); X {X.shape}")
    print("mean BCF %.3f; beats per EA beat median %d; labelled per EA median %d" % (df.BCF.mean(), df.n_total.median(), df.n_labelled.median()))
    print(df.groupby("dataset")[["BCF", "n_surv", "RB_label", "RB_label_sd"]].mean().round(2))
