"""Step 3: Sheikh et al. 2022 feature vector (465 features) per beat, ported from the paper's text + Appendix B.

Time-series features (450): dZ/dt, d2Z/dt2, d3Z/dt3 in [tR, tR+150] ms (151 + 150 + 149 samples at 1 kHz).
High-level (15) from the variable search window  [tR+35, min(tR+140, t(0.66*C amp before C))] ms:
 451 last notch amp, 452 last inflection amp, 453 max d3Z/dt3 (in window), 454 C amp, 455 C-notch, 456 C-inflection,
 457 C-max3D, 458 tLN-tLI, 459 tRC-tLN, 460 tRC-tLI, 461 tRC-t3Dmax, 462 RC, 463 tLN, 464 tLI, 465 t3Dmax.
Missing inflection/notch -> 0 (as in the paper).  Conventions chosen by us where the text is ambiguous are marked 'ASSUMPTION'.
Since our beats are single (not ensemble averaged) and filtered at 40 Hz, derivatives are taken on the filtered signal.
"""
import numpy as np, pandas as pd
from common import *

def feats_one(w):
    """w: ICG window, index j <-> R-100+j ms. Returns (465,) float array and a dict of debugging values."""
    s = w.astype(float); r0 = PRE                             # index of R
    d1 = np.gradient(s); d2 = np.gradient(d1); d3 = np.gradient(d2)   # derivatives per ms ("d2Z/dt2" in paper notation = 1st deriv of ICG)
    # C point: max of ICG in R..R+350 ms  (ASSUMPTION: Sheikh uses the max of the beat; we limit to R+350 for single beats)
    c = r0 + int(np.argmax(s[r0 + 35: r0 + 351])) + 35; camp = s[c]
    lo = r0 + 35
    # upper limit = min(R+140, last time before C where signal <= 0.66*C amp ... i.e. t(0.66*Camp) on the rising limb before C)
    below = np.where(s[lo:c + 1] <= 0.66 * camp)[0]
    t66 = lo + below[-1] if len(below) else lo
    hi = min(r0 + 140, t66) if t66 > lo else r0 + 140          # ASSUMPTION: if 0.66*C is never undercut after lo, fall back to R+140
    hi = max(hi, lo + 5)
    win = np.arange(lo, hi + 1)
    # last inflection: dZ/dt slope positive (d1>0) and d2 (= d3Z/dt3 in paper notation) changes sign - to + ; we use sign change of d2 (neg->pos)
    infl = [i for i in win[1:] if d1[i] > 0 and d2[i - 1] < 0 <= d2[i]]
    notch = [i for i in win[1:-1] if s[i] < s[i - 1] and s[i] <= s[i + 1] and s[i] < 0.0 + 1e9]  # ASSUMPTION: notch = local minimum of ICG in window
    t3 = win[int(np.argmax(d2[win]))]                           # max of "third derivative" (second derivative of ICG)
    tLI = infl[-1] if infl else None; tLN = notch[-1] if notch else None
    ms = lambda i: (i - r0)                                      # time after R in ms
    amp_LN = s[tLN] if tLN is not None else 0.0; amp_LI = s[tLI] if tLI is not None else 0.0
    t_LN = ms(tLN) if tLN is not None else 0.0; t_LI = ms(tLI) if tLI is not None else 0.0
    t_RC = ms(c); t_3 = ms(t3)
    ts = np.concatenate([s[r0: r0 + 151], d1[r0: r0 + 150], d2[r0: r0 + 149]])   # 151+150+149 = 450
    hl = np.array([amp_LN, amp_LI, d2[t3], camp, camp - amp_LN, camp - amp_LI, camp - d2[t3],
                   t_LN - t_LI, t_RC - t_LN, t_RC - t_LI, t_RC - t_3, t_RC, t_LN, t_LI, t_3])
    return np.concatenate([ts, hl]), dict(c=ms(c), hi=ms(hi), has_infl=tLI is not None, has_notch=tLN is not None)

def build(W):
    X = np.empty((len(W), 465)); aux = []
    for i in range(len(W)):
        X[i], a = feats_one(W[i]); aux.append(a)
    return X, pd.DataFrame(aux)

NAMES = [f"icg_{k}" for k in range(151)] + [f"d1_{k}" for k in range(150)] + [f"d2_{k}" for k in range(149)] + [
    "amp_LN", "amp_LI", "amp_max3D", "amp_C", "dAmp_C_LN", "dAmp_C_LI", "dAmp_C_3D", "dt_LN_LI", "dt_RC_LN", "dt_RC_LI", "dt_RC_3D", "RC", "t_LN", "t_LI", "t_3D"]

if __name__ == "__main__":
    m, W, E = load_beats(cohort=True)
    X, aux = build(W)
    assert X.shape[1] == 465 and np.isfinite(X).all()
    np.save(PROC / "sheikh_X.npy", X); m.to_csv(PROC / "cohort_meta.csv", index=False); aux.to_csv(PROC / "sheikh_aux.csv", index=False)
    print("features", X.shape, "has_infl %.3f has_notch %.3f" % (aux.has_infl.mean(), aux.has_notch.mean()), "window hi mean %.1f ms" % aux.hi.mean())
