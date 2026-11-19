"""Step 1: build per-beat table + 1 kHz ICG/ECG windows (R-100..R+600 ms) for both datasets.
Outputs: data/processed/beats_meta.csv, beats_icg.npy, beats_ecg.npy, build_report.json
Only uses information in the files; nothing is imputed."""
import json, sys
import numpy as np, pandas as pd, bioread
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from common import *

ORIG = {"Pause": "Ruhe", "Valsalva": "Valsalva", "HoldingBreath": "AtmungAnhalten", "TiltUp": "TiltUp", "TiltLevel": "TiltDown"}
EMP_PHASES = ["prep", "pause_1", "talk", "math", "pause_5"]
T = np.arange(-PRE, POST + 1) / 1000.0  # window grid (s) relative to R

def process_segment(ecg_raw, icg_raw, fs, lab1_ecg, lab1_icg, lab2_ecg, lab2_icg, meta):
    """ecg_raw/icg_raw: arrays at fs. label frames from read_labels (abs idx into these arrays)."""
    bad = np.isnan(icg_raw) | np.isnan(ecg_raw)           # gaps (kept at original positions, like PEPbench)
    if bad.any():
        idx = np.arange(len(icg_raw))
        icg_raw = np.interp(idx, idx[~bad], icg_raw[~bad]); ecg_raw = np.interp(idx, idx[~bad], ecg_raw[~bad])
    badc = np.concatenate([[0], np.cumsum(bad)])
    icg = icg_filter(icg_raw.astype(float), fs)
    ecg = ecg_filter(ecg_raw.astype(float), fs)
    spl_i, spl_e = CubicSpline(np.arange(len(icg)), icg), CubicSpline(np.arange(len(ecg)), ecg)
    hb = lab1_icg[["start", "end"]].copy()
    hb["B1"] = lab1_icg["point"]; hb["Q1"] = lab1_ecg["point"].reindex(hb.index)
    # rater 2 has its own heartbeat ids -> match by position: the single rater-2 point inside rater-1 beat [start,end)
    def inside(lab, hb):
        if lab is None: return pd.Series(np.nan, index=hb.index)
        p = np.sort(lab["point"].dropna().values)
        lo, hi = np.searchsorted(p, hb["start"].values, "left"), np.searchsorted(p, hb["end"].values, "left")
        return pd.Series([p[a] if b - a == 1 else np.nan for a, b in zip(lo, hi)], index=hb.index)
    hb["B2"] = inside(lab2_icg, hb); hb["Q2"] = inside(lab2_ecg, hb)
    rows, W_i, W_e = [], [], []
    # R peak: argmax of cleaned ECG in [Q-10ms, Q+80ms] (rater-1 Q, else rater-2 Q)
    for hid, r in hb.iterrows():
        q = r.Q1 if not np.isnan(r.Q1) else r.Q2
        if np.isnan(q): continue
        a, b = int(q - 0.010 * fs), int(q + 0.080 * fs)
        if a < 0 or b >= len(ecg): continue
        R = a + int(np.argmax(ecg[a:b + 1]))
        rows.append(dict(meta, heartbeat_id=hid, R=R, start=r.start, end=r.end, B1=r.B1, Q1=r.Q1, B2=r.B2, Q2=r.Q2))
    df = pd.DataFrame(rows)
    if df.empty: return df, None, None
    # RR (ms) to previous beat when consecutive ids
    df["RR_prev_ms"] = np.where(df.heartbeat_id.diff() == 1, df.R.diff() / fs * 1000, np.nan)
    df["RR_next_ms"] = df.RR_prev_ms.shift(-1)
    keep = []
    for k, r in df.iterrows():
        tt = r.R + T * fs
        if tt[0] < 0 or tt[-1] > len(icg) - 1: keep.append(False); continue
        lo, hi = int(tt[0]), int(tt[-1]) + 1
        if badc[hi + 1 if hi + 1 < len(badc) else -1] - badc[lo] > 0: keep.append(False); continue   # window touches NaN gap
        W_i.append(spl_i(tt)); W_e.append(spl_e(tt)); keep.append(True)
    df = df[np.array(keep)].reset_index(drop=True)
    for c in ["B1", "Q1", "B2", "Q2"]:
        df[c + "_ms"] = (df[c] - df.R) / fs * 1000.0       # relative to R peak, ms
    df["fs"] = fs
    return df, np.array(W_i, dtype=np.float32), np.array(W_e, dtype=np.float32)

def lab(path):
    return read_labels(path) if path.exists() else None

def do_guardian(pid):
    out = []
    base = GDN / "data_raw" / pid / "tfm_data"
    m = loadmat(base / f"{pid.lower()}_no01.mat", struct_as_record=False, squeeze_me=True)
    names = list(m["IV"].Name); raw = m["RAW_SIGNALS"]
    for ph, orig in ORIG.items():
        if orig not in names: continue
        idx = names.index(orig)
        try:
            ecg = np.asarray(raw.rawECG2[idx], float); icg = np.asarray(raw.rawICG[idx], float)
        except Exception: continue
        if len(ecg) == 0 or len(icg) == 0: continue
        n = min(len(ecg) // 2, len(icg)); ecg_nan = np.isnan(ecg[: 2 * n]).reshape(n, 2).any(1)
        e = ecg[: 2 * n].copy(); ix = np.arange(len(e)); nn = np.isnan(e)
        if nn.any(): e[nn] = np.interp(ix[nn], ix[~nn], e[~nn])
        ecg = resample_poly(e, 1, 2); ecg[ecg_nan] = np.nan; icg = icg[:n]
        ld = base / "reference_labels"
        f = lambda r, ch: lab(ld / f"rater_0{r}" / f"reference_labels_{pid.lower()}_{ph.lower()}_{ch}.csv")
        l1i, l1e = f(1, "icg"), f(1, "ecg")
        if l1i is None or l1e is None: continue
        meta = dict(dataset="Guardian", subject=pid, condition="", phase=ph)
        out.append(process_segment(ecg, icg, 500.0, l1e, l1i, f(2, "ecg"), f(2, "icg"), meta))
    return out

def do_empkins(vp, cond):
    out = []
    base = EMP / "data_per_subject" / vp / cond / "biopac"
    acq = sorted((base / "raw").glob("*.acq"))
    acq = [a for a in acq if a.name.lower() == f"biopac_data_{vp.lower()}_{cond}.acq"] or acq
    d = bioread.read_file(str(acq[0]))
    ch = {c.name: c.data for c in d.channels}
    ecg = np.asarray(ch["ECG - ECG100C"], float); icg = np.asarray(ch["ICG - Derivative, NICO100C"], float)
    ld = base / "reference_labels"
    for ph in EMP_PHASES:
        f = lambda r, c: lab(ld / f"rater_0{r}" / f"reference_labels_{vp.lower()}_{cond}_{ph}_{c}.csv")
        l1i, l1e = f(1, "icg"), f(1, "ecg")
        if l1i is None or l1e is None: continue
        meta = dict(dataset="EmpkinS", subject=vp, condition=cond, phase=ph)
        out.append(process_segment(ecg, icg, 1000.0, l1e, l1i, f(2, "ecg"), f(2, "icg"), meta))
    return out

if __name__ == "__main__":
    jobs = [("g", p.name) for p in sorted((GDN / "data_raw").iterdir()) if p.is_dir()]
    jobs += [("e", (v.name, c)) for v in sorted((EMP / "data_per_subject").iterdir()) if v.is_dir() for c in ["tsst", "ftsst"]]
    def run(j):
        try:
            return do_guardian(j[1]) if j[0] == "g" else do_empkins(*j[1])
        except Exception as e:
            return [("ERROR", str(j), repr(e))]
    res = Parallel(n_jobs=30)(delayed(run)(j) for j in jobs)
    dfs, Wi, We, errs = [], [], [], []
    for r in res:
        for item in r:
            if isinstance(item[0], str): errs.append(item[1:]); continue
            df, wi, we = item
            if wi is None: continue
            dfs.append(df); Wi.append(wi); We.append(we)
    meta = pd.concat(dfs, ignore_index=True)
    meta.to_csv(PROC / "beats_meta.csv", index=False)
    np.save(PROC / "beats_icg.npy", np.concatenate(Wi)); np.save(PROC / "beats_ecg.npy", np.concatenate(We))
    rep = dict(n_beats=len(meta), by_dataset=meta.groupby("dataset").size().to_dict(),
               n_B1=int(meta.B1.notna().sum()), n_B2=int(meta.B2.notna().sum()), errors=errs)
    json.dump(rep, open(PROC / "build_report.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
