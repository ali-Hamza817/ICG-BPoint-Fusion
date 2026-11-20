"""Step 14: v3 evaluation with the REAL Abel features (step 13) and the REAL Sheikh feature code (step 3c) + Sheikh-style AdaBoost.
Resumable: one pickle per task in results/tasks_v3/.  Protocol:  subject-wise grouped 5-fold CV (x3 repeats for cheap models, x1 for AdaBoost),
hyper-parameters chosen ONLY by inner 3-fold GroupKFold on the training part; cross-dataset = train on one dataset only, test on the other (all models reported, none selected on target).
Models
  abel_rf    : Abel 2026 design: 12 PEPbench rules + RR, NaN kept, RandomForest, target = B rel. heartbeat start (Abel convention)
  abel_svr   : same features, in-cycle median imputation (Abel exp. 2), MinMax + SVR
  sheikh_*   : Sheikh 465 features (his MATLAB port), target RB ;  ada = StandardScaler + AdaBoost(exponential, 500 trees, tree max_features 0.7, depth/lr tuned in
               his published range) + his featureless-beat rule (no notch & no inflection -> prediction = RB_max_2D, as in his notebook)
  fus_*      : Sheikh 465 + Abel 13 (Abel features imputed for ada/svr/ridge; NaN kept for rf/lgbm)
All errors are evaluated in ms against the rater-1 B-point, after converting predictions to ms-after-R."""
import os, pickle, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from common import *

TAG = os.environ.get("TASKS_TAG", "v3")                      # v3 = results before the artefact fix (kept), v4 = after
TASKS = RES / f"tasks_{TAG}"; TASKS.mkdir(exist_ok=True, parents=True)
N_REPEATS, N_FOLDS = 3, 5
ABEL_NAMES = ["arbol2017-isoelectric-crossings", "arbol2017-second-derivative", "arbol2017-third-derivative", "debski1993-second-derivative", "drost2022", "forouzanfar2018",
              "lozano2007-linear-regression", "lozano2007-quadratic-regression", "miljkovic2022", "pale2021", "sherwood1990", "stern1985"]


def load_v3():
    meta = read_meta_clean(); coh = (meta.B1_ms.notna() & (meta.B1_ms - meta.Q1_ms).between(20, 250)).values
    m = pd.read_csv(PROC / "cohort_meta.csv"); assert coh.sum() == len(m)
    AX = pd.read_csv(PROC / "abel_real_X.csv")[coh].reset_index(drop=True)
    S = np.load(PROC / "sheikh_v2_X.npy"); aux = pd.read_csv(PROC / "sheikh_v2_aux.csv")
    start_to_R = AX.start_to_R_ms.values                                      # (R - start) in ms, per beat
    feats = AX[ABEL_NAMES].values.astype(float)
    ok = ~np.isnan(feats).all(axis=1) & AX.rr_interval_ms.notna().values | ~np.isnan(feats).all(axis=1)   # >=1 rule available (Abel excludes beats where all rules fail)
    rr = AX.rr_interval_ms.copy(); rr = rr.fillna(m.assign(rr=rr).groupby("subject").rr.transform("median")).values
    A_nan = np.column_stack([feats, rr])
    imp = pd.DataFrame(feats); imp = imp.apply(lambda c: c.fillna(imp.median(axis=1))).values
    A_imp = np.column_stack([imp, rr])
    FS = {"A13": A_nan, "A13i": A_imp, "S465": S, "FUS": np.hstack([S, A_imp]), "FUSn": np.hstack([S, A_nan])}
    fless = (~aux.has_infl.values) & (~aux.has_notch.values); fb_rb = S[:, -1] - 1.0           # RB_max_2D (MATLAB index) -> ms after R
    keep = ~np.isnan(feats).all(axis=1) & np.isfinite(A_imp).all(axis=1)
    return m, FS, start_to_R, fless, fb_rb, keep


def make_folds(groups, sizes, seed):
    rng = np.random.default_rng(seed); subs = list(rng.permutation(np.array(sorted(set(groups))))); load = np.zeros(N_FOLDS); fo = {}
    for s in sorted(subs, key=lambda s: -sizes[s] + rng.normal(0, 30)):
        f = int(np.argmin(load)); fo[s] = f; load[f] += sizes[s]
    return np.array([fo[g] for g in groups])


def cfgs(model):
    if model in ("abel_rf", "sheikh_rf", "fus_rf"): return [dict(mf=mf, leaf=lf) for mf in (0.3, 0.6, 1.0) for lf in (1, 5)]
    if model in ("sheikh_lgbm", "fus_lgbm"): return [dict(nl=nl, lr=lr) for nl in (15, 31) for lr in (0.03, 0.1)]
    if model in ("abel_svr", "sheikh_svr", "fus_svr"): return [dict(C=C) for C in (1, 10, 100)]
    if model in ("sheikh_ridge", "fus_ridge"): return [dict(alpha=a) for a in (1, 10, 100, 1000, 10000)]
    if model in ("sheikh_ada", "fus_ada"): return [dict(lr=lr, depth=d) for lr in (0.1, 1.0) for d in (12, 20)]


def build(model, c, seed, n_ada=500):
    if model.endswith("_rf"): return RandomForestRegressor(300, max_features=c["mf"], min_samples_leaf=c["leaf"], n_jobs=1, random_state=seed)
    if model.endswith("_lgbm"): return lgb.LGBMRegressor(n_estimators=400, learning_rate=c["lr"], num_leaves=c["nl"], colsample_bytree=0.3, subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1, random_state=seed)
    if model.endswith("_svr"): return make_pipeline(MinMaxScaler(), SVR(C=c["C"], epsilon=1.0, gamma="scale"))
    if model.endswith("_ridge"): return make_pipeline(StandardScaler(), Ridge(alpha=c["alpha"]))
    if model.endswith("_ada"):
        return make_pipeline(StandardScaler(), AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=c["depth"], max_features=0.7, random_state=10),
                                                                 loss="exponential", n_estimators=n_ada, learning_rate=c["lr"], random_state=1))


def fit_predict(model, X, y, g, Xte, seed):
    cs = cfgs(model); best, bm = cs[0], 1e18
    n_inner = 100 if model.endswith("_ada") else 500
    for c in cs:
        errs = []
        for tr, va in GroupKFold(3).split(X, y, g):
            errs.append(np.abs(build(model, c, seed, 100).fit(X[tr], y[tr]).predict(X[va]) - y[va]).mean())
        if np.mean(errs) < bm: bm, best = np.mean(errs), c
    return build(model, best, seed, 500).fit(X, y).predict(Xte), best


FSET_OF = {"abel_rf": "A13", "abel_svr": "A13i", "sheikh_rf": "S465", "sheikh_lgbm": "S465", "sheikh_svr": "S465", "sheikh_ridge": "S465", "sheikh_ada": "S465",
           "fus_rf": "FUSn", "fus_lgbm": "FUSn", "fus_svr": "FUS", "fus_ridge": "FUS", "fus_ada": "FUS"}
START_REL = {"abel_rf", "abel_svr"}                                              # Abel convention: target = B rel. heartbeat start
M, FS, S2R, FLESS, FB_RB, KEEP = load_v3()


def run_task(key, kind, model, rep, fold):
    out = TASKS / f"{key}.pkl"
    if out.exists(): return key
    X = FS[FSET_OF[model]]; rb = M.B1_ms.values; y = rb + S2R if model in START_REL else rb
    idx = np.where(KEEP)[0]; grp = M.subject.values; t0 = time.time()
    if kind == "pooled":
        sizes = M[KEEP].groupby("subject").size().to_dict(); f = make_folds(grp[idx], sizes, rep)
        te = idx[f == fold]; tr = idx[f != fold]
    else:
        src, dst = ("EmpkinS", "Guardian") if kind == "x_e2g" else ("Guardian", "EmpkinS")
        ds = M.dataset.values; tr = idx[ds[idx] == src]; te = idx[ds[idx] == dst]
    p, best = fit_predict(model, X[tr], y[tr], grp[tr], X[te], seed=rep)
    pred = p - S2R[te] if model in START_REL else p                              # -> ms after R
    d = dict(key=key, kind=kind, model=model, rep=rep, fold=fold, test_idx=te, pred_rb=pred, best=best, sec=time.time() - t0)
    if model.endswith("_ada"): d["pred_rb_rule"] = np.where(FLESS[te], FB_RB[te], pred)         # Sheikh's featureless-beat rule
    pickle.dump(d, open(out, "wb")); return key


def task_list():
    cheap = ["abel_rf", "abel_svr", "sheikh_rf", "sheikh_lgbm", "sheikh_svr", "sheikh_ridge", "fus_rf", "fus_lgbm", "fus_svr", "fus_ridge"]; T = []
    for mo in ["sheikh_ada", "fus_ada"]:
        for fo in range(N_FOLDS): T.append(("pooled", mo, 0, fo))
        for k in ("x_e2g", "x_g2e"): T.append((k, mo, 0, 0))
    for mo in cheap:
        for rep in range(N_REPEATS):
            for fo in range(N_FOLDS): T.append(("pooled", mo, rep, fo))
        for k in ("x_e2g", "x_g2e"): T.append((k, mo, 0, 0))
    return [(f"{k}__{mo}__r{r}f{f}", k, mo, r, f) for k, mo, r, f in T]


if __name__ == "__main__":
    T = task_list(); todo = [t for t in T if not (TASKS / f"{t[0]}.pkl").exists()]
    print(f"{len(T)} tasks, {len(todo)} to do; evaluation set n={int(KEEP.sum())} beats", flush=True)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 40; done = 0
    for _ in Parallel(n_jobs=n_jobs, return_as="generator_unordered", backend="loky")(delayed(run_task)(*t) for t in todo):
        done += 1
        if done % 5 == 0: print(f"{time.strftime('%H:%M:%S')} done {done}/{len(todo)}", flush=True)
    print("ALL V3 TASKS DONE", flush=True)
