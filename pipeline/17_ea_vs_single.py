"""Step 17 (advisor direction 3): single beat vs ensemble-averaged beat, same data, same windows, same labels.
For every EA window (step 16) the reference is the EA label = mean rater-1 RB of the surviving labelled beats.
  A) 'EA model'      : model trained AND tested on EA beats (Sheikh's setting; EA features, subject-wise nested CV, 5 folds x 3 repeats)
  B) 'single->avg'   : single-beat models (step 14, subject-wise pooled CV predictions) -> average of the predictions of the beats in the window -> compare with the EA label
  C) 'single (ref)'  : mean per-beat absolute error of the same single-beat models on the same beats (what Abel-style evaluation measures)
All errors in ms.  usage: python 17_ea_vs_single.py <window seconds>"""
import importlib.util, pickle, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from common import *

import os
TAGV = os.environ.get("TASKS_TAG", "v3"); W = int(sys.argv[1]) if len(sys.argv) > 1 else 20; TAG = f"_w{W}"
spec = importlib.util.spec_from_file_location("m14", Path(__file__).with_name("14_models_v3.py")); m14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m14)
N_FOLDS = 5
E = pd.read_csv(PROC / f"ea_meta{TAG}.csv"); XE = np.load(PROC / f"ea_X{TAG}.npy"); y = E.RB_label.values; grp = E.subject.values
fless = (~E.has_infl.values) & (~E.has_notch.values); fb = E.RB_max_2D.values
print(f"window {W}s: {len(E)} EA beats, {E.subject.nunique()} subjects, featureless {fless.mean():.3f}", flush=True)

# ---------- A) EA models ----------
EA_FS = {"sheikh_ridge": XE, "sheikh_svr": XE, "sheikh_rf": XE, "sheikh_lgbm": XE, "sheikh_ada": XE}
res = {}
sizes = E.groupby("subject").size().to_dict()
for model, X in EA_FS.items():
    P = []; Pr = []
    for rep in range(3):
        f = m14.make_folds(grp, sizes, rep); p = np.full(len(E), np.nan)
        for fo in range(N_FOLDS):
            te = np.where(f == fo)[0]; tr = np.where(f != fo)[0]
            if len(set(grp[tr])) < 3: continue
            p[te], _ = m14.fit_predict(model, X[tr], y[tr], grp[tr], X[te], seed=rep)
        P.append(p)
        print(f"  EA {model} rep{rep} MAE {np.nanmean(np.abs(p - y)):.2f}", flush=True)
    res[model] = np.mean(P, axis=0)
    if model == "sheikh_ada": res["sheikh_ada+rule"] = np.where(fless, fb, res[model])

# ---------- B/C) single-beat predictions aggregated per window ----------
meta = read_meta_clean(); coh = (meta.B1_ms.notna() & (meta.B1_ms - meta.Q1_ms).between(20, 250)).values
pos = np.full(len(meta), -1); pos[coh] = np.arange(coh.sum()); Mc = pd.read_csv(PROC / "cohort_meta.csv"); yb = Mc.B1_ms.values
single = {}
for mo in sorted({d["model"] for d in (pickle.load(open(f, "rb")) for f in glob.glob(str(RES / f"tasks_{TAGV}/pooled__*.pkl")))}):
    ps = {}
    for f in glob.glob(str(RES / f"tasks_{TAGV}/pooled__{mo}__r*.pkl")):
        d = pickle.load(open(f, "rb")); ps.setdefault(d["rep"], []).append(d)
    arrs = []; arrs_rule = []
    for rep, ds in ps.items():
        if len(ds) != 5: continue
        p = np.full(len(Mc), np.nan); pr = p.copy()
        for d in ds: p[d["test_idx"]] = d["pred_rb"]; pr[d["test_idx"]] = d.get("pred_rb_rule", d["pred_rb"])
        arrs.append(p); arrs_rule.append(pr)
    if arrs: single[mo] = np.mean(arrs, axis=0); single[mo + "+rule"] = np.mean(arrs_rule, axis=0) if mo.endswith("_ada") else None
rows = []
rowsets = [[int(x) for x in s.split(";")] for s in E.beat_rows]
for mo, p in single.items():
    if p is None: continue
    agg = np.full(len(E), np.nan); per = np.full(len(E), np.nan)
    for i, rs in enumerate(rowsets):
        ci = pos[rs]; ci = ci[ci >= 0]; ci = ci[~np.isnan(p[ci])]
        if len(ci) >= 3: agg[i] = p[ci].mean(); per[i] = np.abs(p[ci] - yb[ci]).mean()
    res["single->avg:" + mo] = agg; res["__per_beat_" + mo] = per

def summarize(name, pred, ref=y):
    e = np.abs(pred - ref); k = ~np.isnan(e)
    if k.sum() < 10: return None
    sj = pd.Series(e[k]).groupby(grp[k]).mean()
    return dict(window_s=W, approach=name, n_windows=int(k.sum()), n_subjects=int(sj.shape[0]), MAE=e[k].mean(), SD=e[k].std(), MedAE=np.median(e[k]), subjectMAE=sj.mean())
for name, p in res.items():
    if name.startswith("__per_beat_"):
        e = p; k = ~np.isnan(e); sj = pd.Series(e[k]).groupby(grp[k]).mean()
        rows.append(dict(window_s=W, approach="single-beat error (mean |pred-label| per beat): " + name[11:], n_windows=int(k.sum()), n_subjects=int(sj.shape[0]), MAE=e[k].mean(), SD=np.nan, MedAE=np.nan, subjectMAE=sj.mean()))
    else:
        r = summarize(("EA model: " if not name.startswith("single") else "") + name, p)
        if r: rows.append(r)
T = pd.DataFrame(rows); T.round(3).to_csv(RES / f"ea_vs_single_{TAGV}{TAG}.csv", index=False); print(T.round(2).to_string(index=False))
pickle.dump(dict(res=res, E=E), open(RES / f"ea_vs_single_{TAGV}{TAG}_predictions.pkl", "wb"))
