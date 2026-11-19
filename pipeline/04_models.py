"""Step 4: subject-wise (a) pooled 5-fold CV x3 repeats and (b) cross-dataset experiments. Resumable: one pickle per task.
Models: adaboost (Sheikh 2022 design), rf (Abel 2026 design), lgbm, svr, ridge.  Nested tuning: small config list, inner 3-fold GroupKFold.
DEVIATION from Sheikh: tuning over a 4-config SUBSET of his grid (full 180-combo nested grid infeasible: ~1 s/tree)."""
import itertools, pickle, sys, time
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

TASKS = RES / "tasks"; TASKS.mkdir(exist_ok=True, parents=True)
N_REPEATS, N_FOLDS = 3, 5

def load_all():
    m = pd.read_csv(PROC / "cohort_meta.csv"); X = np.load(PROC / "sheikh_X.npy"); R = pd.read_csv(PROC / "rule_X.csv")
    rb = [c for c in R.columns if c != "RC"]
    R[rb] = R[rb].apply(lambda col: col.fillna(R[rb].median(axis=1)))          # in-cycle median imputation (as Abel exp.2)
    rr = m.groupby("subject").RR_prev_ms.transform("median"); rr = m.RR_prev_ms.where(m.RR_prev_ms > 250).fillna(rr)
    camp = np.maximum(X[:, 453], 1e-6)[:, None]
    Xn = X.copy(); Xn[:, :456] = X[:, :456] / camp                                 # amplitude-normalised (scale-free) variant
    rules = np.column_stack([R.values, rr.values])
    FS = {"sheikh465": X, "rules": rules, "fusion": np.hstack([X, rules]), "sheikh465n": Xn, "fusionn": np.hstack([Xn, rules])}
    return m, FS

def make_folds(groups, sizes, seed):
    rng = np.random.default_rng(seed); subs = list(rng.permutation(np.array(sorted(set(groups)))))
    load = np.zeros(N_FOLDS); fold_of = {}
    for s in sorted(subs, key=lambda s: -sizes[s] + rng.normal(0, 30)):               # greedy balance with random tie-break
        f = int(np.argmin(load)); fold_of[s] = f; load[f] += sizes[s]
    return np.array([fold_of[g] for g in groups])

def cfgs(model):
    if model == "adaboost":
        return [dict(lr=lr, depth=d, mf=mf) for lr in (0.1, 0.5) for d, mf in ((5, 0.3), (10, 0.1))]
    if model == "rf": return [dict(mf=mf, depth=d) for mf, d in ((0.1, 10), (0.3, 15), (0.5, None), (0.3, None))]
    if model == "lgbm": return [dict(nl=nl, lr=lr) for nl in (15, 31) for lr in (0.03, 0.1)]
    if model == "svr": return [dict(C=C, eps=e) for C in (1, 10, 100) for e in (1.0,)]
    if model == "ridge": return [dict(alpha=a) for a in (1, 10, 100, 1000, 10000)]

def build(model, c, seed=0):
    if model == "adaboost":
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=c["depth"], max_features=c["mf"], random_state=seed),
                                 n_estimators=500, loss="exponential", learning_rate=c["lr"], random_state=seed)
    if model == "rf": return RandomForestRegressor(300, max_features=c["mf"], max_depth=c["depth"], n_jobs=1, random_state=seed)
    if model == "lgbm": return lgb.LGBMRegressor(n_estimators=400, learning_rate=c["lr"], num_leaves=c["nl"], colsample_bytree=0.3, subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1, random_state=seed)
    if model == "svr": return make_pipeline(MinMaxScaler(), SVR(C=c["C"], epsilon=c["eps"], gamma="scale"))
    if model == "ridge": return make_pipeline(StandardScaler(), Ridge(alpha=c["alpha"]))

def fit_predict(model, X, y, g, Xte, seed):
    """nested: choose config by inner 3-fold GroupKFold MAE on training set, refit on all training data."""
    cs = cfgs(model); best, bm = None, 1e18
    if len(cs) > 1:
        for c in cs:
            errs = []
            for tr, va in GroupKFold(3).split(X, y, g):
                errs.append(np.abs(build(model, c, seed).fit(X[tr], y[tr]).predict(X[va]) - y[va]).mean())
            if np.mean(errs) < bm: bm, best = np.mean(errs), c
    else: best = cs[0]
    return build(model, best, seed).fit(X, y).predict(Xte), best

def run_task(key, kind, model, fset, target, rep, fold):
    out = TASKS / f"{key}.pkl"
    if out.exists(): return key
    m, FS = DATA
    X = FS[fset]; rb = m.B1_ms.values; rc = m.C_ms.values.astype(float)
    y = rb - rc if target == "dRC" else rb
    grp = m.subject.values
    if kind == "pooled":
        sizes = m.groupby("subject").size().to_dict(); f = make_folds(grp, sizes, rep); te = np.where(f == fold)[0]; tr = np.where(f != fold)[0]
    else:
        src, dst = ("EmpkinS", "Guardian") if kind == "x_emp2gdn" else ("Guardian", "EmpkinS")
        tr = np.where(m.dataset == src)[0]; te = np.where(m.dataset == dst)[0]
    t0 = time.time()
    p, best = fit_predict(model, X[tr], y[tr], grp[tr], X[te], seed=rep)
    pred_rb = p + rc[te] if target == "dRC" else p
    pickle.dump(dict(key=key, kind=kind, model=model, fset=fset, target=target, rep=rep, fold=fold, test_idx=te, pred_rb=pred_rb, best=best, sec=time.time() - t0), open(out, "wb"))
    return key

def task_list():
    T = []
    fast = ["rf", "lgbm", "svr", "ridge"]
    for model in fast:
        for fset in ("sheikh465", "rules", "fusion"):
            for tg in ("RB", "dRC"):
                for rep in range(N_REPEATS):
                    for fold in range(N_FOLDS): T.append(("pooled", model, fset, tg, rep, fold))
    for fset in ("sheikh465", "fusion"):
        for tg in ("RB", "dRC"):
            for rep in range(N_REPEATS):
                for fold in range(N_FOLDS): T.append(("pooled", "adaboost", fset, tg, rep, fold))
    for kind in ("x_emp2gdn", "x_gdn2emp"):
        for model in ["adaboost"] + fast:
            for fset in ("sheikh465", "rules", "fusion", "sheikh465n", "fusionn"):
                for tg in ("RB", "dRC"): T.append((kind, model, fset, tg, 0, 0))
    T = [(f"{k}__{mo}__{fs}__{tg}__r{r}f{f}", k, mo, fs, tg, r, f) for k, mo, fs, tg, r, f in T]
    heavy = {"adaboost": 0, "rf": 1, "lgbm": 2, "svr": 3, "ridge": 4}
    return sorted(T, key=lambda t: heavy[t[2]])

DATA = load_all() if __name__ != "__main__" or True else None
if __name__ == "__main__":
    T = task_list(); todo = [t for t in T if not (TASKS / f"{t[0]}.pkl").exists()]
    print(f"{len(T)} tasks, {len(todo)} to do", flush=True)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    done = 0
    for _ in Parallel(n_jobs=n_jobs, return_as="generator_unordered", backend="loky")(delayed(run_task)(*t) for t in todo):
        done += 1
        if done % 10 == 0: print(f"{time.strftime('%H:%M:%S')} done {done}/{len(todo)}", flush=True)
    print("ALL TASKS DONE", flush=True)
