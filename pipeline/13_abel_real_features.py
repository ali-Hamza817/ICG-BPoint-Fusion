"""Step 13: Abel et al. 2026 features computed with Abel's OWN algorithm files (PEPbench -> biopsykit; copied unmodified into
external/biopsykit_shim/..., run through a tiny shim for tpcp/validators).  Replaces the re-implementation in 08_abel_features.py.
Pipeline (as PEPbench): ICG band-pass Butterworth 4th order 0.5-25 Hz (IcgPreprocessingBandpass) on the continuous phase signal at native fs
 -> C-points (CPointExtractionScipyFindPeaks) -> 12 B-point algorithms -> features in ms relative to heartbeat start, + RR interval.
DEVIATIONS from Abel (documented): heartbeats = the annotated reference heartbeats (start/end from labels) with R = our R (argmax of cleaned ECG near
 the annotated Q), not PEPbench's neurokit-segmented+matched heartbeats; beats dropped in step 1 (NaN gap / window) are absent from the sequence
 (neighbour-dependent algorithms, i.e. Forouzanfar + C tracking, see the previous kept beat).  RR interval = |R(i) - R(i+1)| as in Abel's code.
Output: data/processed/abel_real_X.csv (one row per beat of beats_meta.csv, same order) + results/abel_real_rule_mae.csv"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, bioread
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "biopsykit_shim"))
from common import *
from biopsykit.signals.icg.event_extraction._c_point_scipy_findpeaks import CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.event_extraction import _b_point_arbol2017 as A, _b_point_debski1993 as De, _b_point_drost2022 as Dr, _b_point_forouzanfar2018 as Fo
from biopsykit.signals.icg.event_extraction import _b_point_lozano2007 as Lo, _b_point_miljkovic2022 as Mi, _b_point_pale2021 as Pa, _b_point_sherwood1990 as Sh, _b_point_stern1985 as St

ALGOS = {"arbol2017-isoelectric-crossings": A.BPointExtractionArbol2017IsoelectricCrossings, "arbol2017-second-derivative": A.BPointExtractionArbol2017SecondDerivative,
         "arbol2017-third-derivative": A.BPointExtractionArbol2017ThirdDerivative, "debski1993-second-derivative": De.BPointExtractionDebski1993SecondDerivative,
         "drost2022": Dr.BPointExtractionDrost2022, "forouzanfar2018": Fo.BPointExtractionForouzanfar2018,
         "lozano2007-linear-regression": Lo.BPointExtractionLozano2007LinearRegression, "lozano2007-quadratic-regression": Lo.BPointExtractionLozano2007QuadraticRegression,
         "miljkovic2022": Mi.BPointExtractionMiljkovic2022, "pale2021": Pa.BPointExtractionPale2021, "sherwood1990": Sh.BPointExtractionSherwood1990, "stern1985": St.BPointExtractionStern1985}
ORIG = {"Pause": "Ruhe", "Valsalva": "Valsalva", "HoldingBreath": "AtmungAnhalten", "TiltUp": "TiltUp", "TiltLevel": "TiltDown"}
EMP_PHASES = ["prep", "pause_1", "talk", "math", "pause_5"]


def run_segment(icg_raw, fs, hb):
    """icg_raw: native-rate raw ICG (gaps already interpolated); hb: DataFrame start_sample,end_sample,r_peak_sample (RangeIndex)."""
    sos = butter(4, [0.5, 25], btype="bandpass", output="sos", fs=fs)                       # IcgPreprocessingBandpass (biopsykit)
    icg = pd.DataFrame({"icg_der": sosfiltfilt(sos, icg_raw.astype(float))})
    cp = CPointExtractionScipyFindPeaks(handle_missing_events="ignore").extract(icg=icg, heartbeats=hb, sampling_rate_hz=fs).points_
    out = pd.DataFrame(index=hb.index)
    out["C_sample"] = cp["c_point_sample"].astype(float)
    start = hb["start_sample"].astype(float)
    for name, cls in ALGOS.items():
        try:
            a = cls(handle_missing_events="ignore").extract(icg=icg, heartbeats=hb, c_points=cp, sampling_rate_hz=fs)
            b = a.points_["b_point_sample"].astype(float)
        except Exception as e:                                                               # algorithm failed for the whole segment -> NaN (logged)
            b = pd.Series(np.nan, index=hb.index); print("ALGO FAIL", name, repr(e)[:100], flush=True)
        out[name] = (b - start) / fs * 1000                                                  # ms from heartbeat start (Abel convention)
    out["rr_interval_ms"] = hb["r_peak_sample"].astype(float).diff(periods=-1).abs() / fs * 1000   # Abel: RR to next beat
    out["start_to_R_ms"] = (hb["r_peak_sample"] - hb["start_sample"]) / fs * 1000
    return out


def prep(ecg, icg):
    bad = np.isnan(icg) | np.isnan(ecg)
    if bad.any():
        ix = np.arange(len(icg)); icg = np.interp(ix, ix[~bad], icg[~bad])
    return icg


def segments_guardian(pid):
    base = GDN / "data_raw" / pid / "tfm_data"
    m = loadmat(base / f"{pid.lower()}_no01.mat", struct_as_record=False, squeeze_me=True); names = list(m["IV"].Name); raw = m["RAW_SIGNALS"]
    for ph, orig in ORIG.items():
        if orig not in names: continue
        idx = names.index(orig)
        try: ecg = np.asarray(raw.rawECG2[idx], float); icg = np.asarray(raw.rawICG[idx], float)
        except Exception: continue
        if len(ecg) == 0 or len(icg) == 0: continue
        n = min(len(ecg) // 2, len(icg)); ecg_nan = np.isnan(ecg[: 2 * n]).reshape(n, 2).any(1)
        e = ecg[: 2 * n].copy(); ix = np.arange(len(e)); nn = np.isnan(e)
        if nn.any(): e[nn] = np.interp(ix[nn], ix[~nn], e[~nn])
        ecg = resample_poly(e, 1, 2); ecg[ecg_nan] = np.nan; icg = icg[:n]
        yield ("Guardian", pid, "", ph), prep(ecg, icg), 500.0


def segments_empkins(vp, cond):
    base = EMP / "data_per_subject" / vp / cond / "biopac"
    acq = [a for a in sorted((base / "raw").glob("*.acq")) if a.name.lower() == f"biopac_data_{vp.lower()}_{cond}.acq"] or sorted((base / "raw").glob("*.acq"))
    d = bioread.read_file(str(acq[0])); ch = {c.name: c.data for c in d.channels}
    ecg = np.asarray(ch["ECG - ECG100C"], float); icg = prep(ecg, np.asarray(ch["ICG - Derivative, NICO100C"], float))
    for ph in EMP_PHASES: yield ("EmpkinS", vp, cond, ph), icg, 1000.0


def job(j, meta):
    res = []
    try:
        gen = segments_guardian(j[1]) if j[0] == "g" else segments_empkins(*j[1])
        for key, icg, fs in gen:
            g = meta[(meta.dataset == key[0]) & (meta.subject == key[1]) & (meta.condition.fillna("") == key[2]) & (meta.phase == key[3])]
            if len(g) < 5: continue
            g = g.sort_values("heartbeat_id")
            hb = pd.DataFrame({"start_sample": g.start.astype(int).values, "end_sample": g.end.astype(int).values, "r_peak_sample": g.R.astype(int).values})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); o = run_segment(icg, fs, hb)
            o.index = g.index; res.append(o)
    except Exception as e:
        print("JOB FAIL", j, repr(e)[:200], flush=True)
    return res


if __name__ == "__main__":
    meta = read_meta_clean()
    jobs = [("g", p.name) for p in sorted((GDN / "data_raw").iterdir()) if p.is_dir()]
    jobs += [("e", (v.name, c)) for v in sorted((EMP / "data_per_subject").iterdir()) if v.is_dir() for c in ["tsst", "ftsst"]]
    parts = [o for r in Parallel(n_jobs=24, backend="loky")(delayed(job)(j, meta) for j in jobs) for o in r]
    X = pd.concat(parts).reindex(meta.index)
    X.to_csv(PROC / "abel_real_X.csv", index=False)
    names = list(ALGOS); print(f"beats {len(meta)}, with features {X[names].notna().any(axis=1).sum()}")
    m = pd.read_csv(PROC / "cohort_meta.csv"); coh = meta.index[(meta.B1_ms.notna()) & ((meta.B1_ms - meta.Q1_ms).between(20, 250))]
    Xc = X.loc[coh].reset_index(drop=True); assert len(Xc) == len(m)
    y = (meta.loc[coh, "B1"].values - meta.loc[coh, "start"].values) / meta.loc[coh, "fs"].values * 1000          # B1 relative to heartbeat start (Abel target)
    err = Xc[names].sub(y, axis=0).abs()
    t = pd.DataFrame({"MAE_vs_rater1_ms": err.mean().round(2), "coverage": (1 - Xc[names].isna().mean()).round(3)}); print(t)
    t.to_csv(RES / "abel_real_rule_mae.csv")
    print("beats with >=1 missing rule:", int(Xc[names].isna().any(axis=1).sum()), "of", len(Xc))
