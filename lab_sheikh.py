"""In-browser copy of pipeline/03c_sheikh_features_v2.py. Step 3c: LINE-BY-LINE Python port of Sheikh's own MATLAB feature extractor
(external/sheikh_toolbox_B_point/feat_vec_465.m, cliffordlab/ICG_OSToolbox, GPL-3).  Supersedes the paper-text-based port in 03_sheikh_features.py.
MATLAB is 1-based; arrays below are padded so that  X1[m]  is MATLAB index m  (time after R = m-1 ms).
Output order = Sheikh's own: [icg(151) D1(150) D2(149) | mag_RB_val mag_RB_infl mag_RB_max_2D mag_C mag_C_val mag_C_infl mag_C_max_2D
 Diff_val_infl Diff_RC_val Diff_RC_infl Diff_RC_max_2D RC RB_infl RB_val RB_max_2D]  = 465 values (target RB_out is NOT included).
Deviations from the MATLAB file are marked DEVIATION.  Validation: pipeline/12_validate_sheikh_port.py (his two demo beats)."""
import numpy as np
from scipy.signal import find_peaks

HL_NAMES = ["mag_RB_val", "mag_RB_infl", "mag_RB_max_2D", "mag_C", "mag_C_val", "mag_C_infl", "mag_C_max_2D",
            "Diff_val_infl", "Diff_RC_val", "Diff_RC_infl", "Diff_RC_max_2D", "RC", "RB_infl", "RB_val", "RB_max_2D"]
NAMES = [f"icg_{k}" for k in range(151)] + [f"D1_{k}" for k in range(150)] + [f"D2_{k}" for k in range(149)] + HL_NAMES
pad = lambda a: np.concatenate([[np.nan], np.asarray(a, float)])        # 1-based accessor


def sheikh_core(icg151, D1, D2, D2sign, RC, mag_C):
    """icg151: dZ/dt at R+0..R+150 ms; D1/D2/D2sign as in MATLAB (first 150 / 149 samples used); RC: C-point time after R (ms); mag_C: C amplitude."""
    icg = pad(icg151); D1p = pad(D1[:150]); D2p = pad(D2[:149]); sgp = pad(D2sign[:149])
    l_lim = 35; u_lim = 140 if RC >= 140 else int(RC)
    pt6 = 0.66 * mag_C
    neg = np.where(np.sign(pt6 - icg[1:u_lim + 1]) == -1)[0]                 # first sample ABOVE 0.66*C (handles double-peak / notched C)
    u_lim_max = u_lim if len(neg) == 0 else int(neg[0])                      # = pos_to_neg_max(1) - 1   (neg[0] is 0-based -> MATLAB idx neg[0]+1)
    u_lim_max = max(u_lim_max, l_lim + 2)                                    # DEVIATION: MATLAB would crash on an empty window; we keep a 3-sample minimum
    # ---- last inflection ----
    for_Zing = sgp[l_lim:u_lim_max + 1]
    n2p = np.where(np.sign(np.diff(for_Zing)) == 1)[0] + 1                   # MATLAB ids (1-based within for_Zing)
    if len(n2p) == 0:
        RB_infl = mag_RB_infl = mag_C_infl = Diff_RC_infl = 0.0
    else:
        allc = n2p + l_lim - 1
        sel = allc[D1p[allc] >= 0]
        sel = sel[icg[sel] < pt6] if len(sel) else sel
        if len(sel) == 0:
            RB_infl, mag_RB_infl = 0.0, 0.0; Diff_RC_infl = RC - RB_infl; mag_C_infl = mag_C - mag_RB_infl
        else:
            RB_infl = float(sel.max()); mag_RB_infl = icg[int(RB_infl)]; mag_C_infl = mag_C - mag_RB_infl; Diff_RC_infl = RC - RB_infl
    # ---- max of d3Z/dt3 (MATLAB "feat_icg_D2") ----
    rg = D2p[l_lim:u_lim_max + 1]
    RB_max_2D = int(np.argmax(rg)) + 1 + l_lim - 1
    mag_RB_max_2D = icg[RB_max_2D]; mag_C_max_2D = mag_C - mag_RB_max_2D; Diff_RC_max_2D = RC - RB_max_2D
    # ---- last notch: findpeaks(-icg) on MATLAB idx 30..u_lim_max ----
    for_val = icg[30:u_lim_max + 1]
    tr = find_peaks(-for_val)[0] + 1                                         # DEVIATION: scipy marks flat peaks at their middle, MATLAB at the first sample
    RB_val = mag_RB_val = mag_C_val = 0.0
    if len(tr):
        allv = tr + 29 - 1; magv = allv[icg[allv] < pt6]
        if len(magv): RB_val = float(magv.max()); mag_RB_val = icg[int(RB_val)]; mag_C_val = mag_C - mag_RB_val
    Diff_val_infl, Diff_RC_val = (0.0, 0.0) if RB_val == 0 else (RB_val - RB_infl, RC - RB_val)
    hl = [mag_RB_val, mag_RB_infl, mag_RB_max_2D, mag_C, mag_C_val, mag_C_infl, mag_C_max_2D, Diff_val_infl, Diff_RC_val, Diff_RC_infl, Diff_RC_max_2D, RC, RB_infl, RB_val, RB_max_2D]
    return np.concatenate([icg[1:152], D1p[1:], D2p[1:], hl]), dict(u_lim_max=u_lim_max, has_infl=RB_infl > 0, has_notch=RB_val > 0)


def sheikh_from_window(w, pre, RC, mag_C):
    """Our beats: w = ICG window, index j <-> R-pre+j ms. Derivatives = np.gradient (verified against his demo D1: max diff 6e-5)."""
    s = np.asarray(w, float); d1 = np.gradient(s); d2 = np.gradient(d1)
    return sheikh_core(s[pre:pre + 151], d1[pre:pre + 150], d2[pre:pre + 149], np.sign(d2[pre:pre + 149]), RC, mag_C)




def features_live(w404):
    """465 Sheikh features from a 404-sample ICG window (R-2 .. R+401 ms, 1 kHz). R is at index 2. C = max of the ICG in R+50..R+400 ms."""
    s = np.asarray(w404, dtype=np.float64); pre = 2
    seg = s[pre + 50: pre + 401]; c = int(np.argmax(seg)) + 50
    return sheikh_from_window(np.asarray(w404), pre, float(c), float(s[pre + c]))
