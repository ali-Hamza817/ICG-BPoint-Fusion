"""Step 9: FAIR evaluation. Resumable (one pickle per task in results/tasks_fair/).
 (a) pooled subject-wise 5-fold CV x3 repeats (same folds as step 4) for Abel-style rule features vs Sheikh features vs fusion.
 (b) cross-dataset: every candidate (model, feature set, target) is trained on the SOURCE dataset only; the candidate to
     report is chosen by SOURCE-ONLY subject-wise 5-fold CV (never by target error). All candidates are stored and reported.
 Per-subject z-scoring ("_sz") uses only each subject's own unlabeled features (label-free, transductive domain adaptation)."""
import importlib.util, pickle, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *

spec = importlib.util.spec_from_file_location("m04", Path(__file__).with_name("04_models.py"))
m04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m04)
make_folds, fit_predict = m04.make_folds, m04.fit_predict
TASKS = RES / "tasks_fair"; TASKS.mkdir(exist_ok=True, parents=True)
N_REPEATS, N_FOLDS = 3, 5


def subject_z(X, subj):
    Z = np.zeros_like(X, dtype=float)
    for s in np.unique(subj):
        k = subj == s; mu = np.nanmean(X[k], 0); sd = np.nanstd(X[k], 0)
        Z[k] = (X[k] - mu) / np.where(sd > 1e-9, sd, 1.0)
    return Z


def load_fair():
    m, FS = m04.load_all()
    A = pd.read_csv(PROC / "abel11_X.csv"); assert len(A) == len(m)
    A_nan = A.values.astype(float)
    A_imp = A.apply(lambda col: col.fillna(A.median(axis=1))).values.astype(float)      # in-cycle median imputation (Abel exp. 2)
    rr = m.groupby("subject").RR_prev_ms.transform("median"); rr = m.RR_prev_ms.where(m.RR_prev_ms > 250).fillna(rr).values[:, None]
    abel11 = np.hstack([A_imp, rr]); abel11nan = np.hstack([A_nan, rr])
    sh = FS["sheikh465"]; fusion = np.hstack([sh, abel11]); subj = m.subject.values
    F = {"abel11": abel11, "abel11nan": abel11nan, "sheikh465": sh, "fusion_a": fusion,
         "abel11_sz": subject_z(abel11, subj), "sheikh465_sz": subject_z(sh, subj), "fusion_a_sz": subject_z(fusion, subj)}
    return m, F


M, F = load_fair()


def run_task(key, kind, model, fset, target, rep, fold):
    out = TASKS / f"{key}.pkl"
    if out.exists(): return key
    X = F[fset]; rb = M.B1_ms.values; rc = M.C_ms.values.astype(float); y = rb - rc if target == "dRC" else rb
    grp = M.subject.values; t0 = time.time()
    if kind == "pooled":
        sizes = M.groupby("subject").size().to_dict(); f = make_folds(grp, sizes, rep)
        te = np.where(f == fold)[0]; tr = np.where(f != fold)[0]
    elif kind.startswith("srccv_"):                               # subject-wise CV inside the SOURCE dataset only
        src = {"e2g": "EmpkinS", "g2e": "Guardian"}[kind[6:]]
        idx = np.where(M.dataset.values == src)[0]; sizes = M.iloc[idx].groupby("subject").size().to_dict()
        f = make_folds(grp[idx], sizes, 0); te = idx[f == fold]; tr = idx[f != fold]
    else:                                                          # x_e2g / x_g2e: train on source, test on target
        src, dst = ("EmpkinS", "Guardian") if kind == "x_e2g" else ("Guardian", "EmpkinS")
        tr = np.where(M.dataset.values == src)[0]; te = np.where(M.dataset.values == dst)[0]
    p, best = fit_predict(model, X[tr], y[tr], grp[tr], X[te], seed=rep)
    pred = p + rc[te] if target == "dRC" else p
    pickle.dump(dict(key=key, kind=kind, model=model, fset=fset, target=target, rep=rep, fold=fold, test_idx=te,
                     pred_rb=pred, best=best, sec=time.time() - t0), open(out, "wb"))
    return key


def task_list():
    T = []; models = ["rf", "svr", "ridge", "lgbm"]
    for model in models:
        for fset in ("abel11", "fusion_a", "fusion_a_sz"):
            for tg in ("RB", "dRC"):
                for rep in range(N_REPEATS):
                    for fold in range(N_FOLDS): T.append(("pooled", model, fset, tg, rep, fold))
    for rep in range(N_REPEATS):                                   # Abel replica: RF on rule features with missing values kept (Exp. 1)
        for fold in range(N_FOLDS): T.append(("pooled", "rf", "abel11nan", "RB", rep, fold))
    for d in ("e2g", "g2e"):
        for model in models:
            for fset in ("abel11", "sheikh465", "fusion_a", "abel11_sz", "sheikh465_sz", "fusion_a_sz"):
                for tg in ("RB", "dRC"):
                    T.append((f"x_{d}", model, fset, tg, 0, 0))
                    for fold in range(N_FOLDS): T.append((f"srccv_{d}", model, fset, tg, 0, fold))
        for fold in range(N_FOLDS): T.append((f"srccv_{d}", "rf", "abel11nan", "RB", 0, fold))
        T.append((f"x_{d}", "rf", "abel11nan", "RB", 0, 0))
    T = [(f"{k}__{mo}__{fs}__{tg}__r{r}f{f}", k, mo, fs, tg, r, f) for k, mo, fs, tg, r, f in T]
    order = {"rf": 0, "lgbm": 1, "svr": 2, "ridge": 3}
    return sorted(T, key=lambda t: order[t[2]])


if __name__ == "__main__":
    T = task_list(); todo = [t for t in T if not (TASKS / f"{t[0]}.pkl").exists()]
    print(f"{len(T)} tasks, {len(todo)} to do", flush=True)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 48; done = 0
    for _ in Parallel(n_jobs=n_jobs, return_as="generator_unordered", backend="loky")(delayed(run_task)(*t) for t in todo):
        done += 1
        if done % 10 == 0: print(f"{time.strftime('%H:%M:%S')} done {done}/{len(todo)}", flush=True)
    print("ALL FAIR TASKS DONE", flush=True)
