"""Shared paths + loaders for EmpkinS / Guardian (own implementation; PEPbench code is NOT executed)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
import neurokit2 as nk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
EMP = DATA / "Empkins_Dataset"
GDN = DATA / "Guardian_Dataset"

# window stored per beat on a 1 kHz grid: R-PRE .. R+POST ms
PRE, POST = 100, 600
GUARDIAN_PHASES = {"Pause": 1, "Valsalva": 3, "HoldingBreath": 4, "TiltUp": 5, "TiltLevel": 6}  # name -> raw array index

def icg_filter(x, fs):
    """4th-order Butterworth band-pass 0.5-40 Hz, zero-phase (as in Sheikh et al. 2022)."""
    sos = butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def ecg_filter(x, fs):
    return np.asarray(nk.ecg_clean(x, sampling_rate=int(fs), method="neurokit"))

def read_labels(path):
    """-> DataFrame heartbeat_id, start, end, point (abs sample idx) for one rater/phase/channel file."""
    d = pd.read_csv(path)
    hb = d[d.channel == "heartbeat"].pivot(index="heartbeat_id", columns="label", values="sample_absolute")
    pt = d[d.channel != "heartbeat"].set_index("heartbeat_id")["sample_absolute"].rename("point")
    return hb[["start", "end"]].join(pt)


def load_beats(cohort=True):
    """-> meta DataFrame, ICG windows (n, 701; column j = R-100+j ms), ECG windows. Adds PEP, HR, C-point."""
    m = pd.read_csv(PROC / "beats_meta.csv")
    W = np.load(PROC / "beats_icg.npy"); E = np.load(PROC / "beats_ecg.npy")
    m["PEP1"] = m.B1_ms - m.Q1_ms
    m["PEP2"] = m.B2_ms - m.Q2_ms
    m["in_cohort"] = m.B1_ms.notna() & m.PEP1.between(20, 250)
    seg = W[:, PRE + 50: PRE + 401]                       # C search window R+50..R+400 ms
    m["C_ms"] = seg.argmax(1) + 50
    m["C_amp"] = seg.max(1)
    m["HR"] = 60000.0 / m.RR_prev_ms.where(m.RR_prev_ms > 250)
    m["row"] = np.arange(len(m))
    if cohort:
        k = m.in_cohort.values
        return m[k].reset_index(drop=True), W[k], E[k]
    return m, W, E
