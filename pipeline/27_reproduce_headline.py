"""Step 27: reproduction check. Re-trains the headline model (Sheikh 465 features + AdaBoost, 500 trees) FROM SCRATCH, using only the recorded settings and the saved test
indices, and compares the new predictions with the stored ones. 5 pooled folds + the two cross-dataset fits = 7 independent fits.
Output: logs/reproduction_headline.log (printed), results/reproduction_headline.json"""
import importlib.util, json, os, pickle, sys, time, glob
from pathlib import Path
import numpy as np
os.environ["TASKS_TAG"] = "v4"
ROOT = Path(__file__).resolve().parents[1]
KEYS = [f"pooled__sheikh_ada__r0f{k}" for k in range(5)] + ["x_e2g__sheikh_ada__r0f0", "x_g2e__sheikh_ada__r0f0"]
def load():
    spec = importlib.util.spec_from_file_location("m14", ROOT / "pipeline" / "14_models_v3.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def fit_one(key):
    m14 = load(); d = pickle.load(open(ROOT / "results" / "tasks_v4" / f"{key}.pkl", "rb"))
    idx = np.where(m14.KEEP)[0]; te = d["test_idx"]; ds = m14.M.dataset.values
    if d["kind"] == "pooled": tr = np.setdiff1d(idx, te)
    else: src = "EmpkinS" if d["kind"] == "x_e2g" else "Guardian"; tr = idx[ds[idx] == src]
    X = m14.FS[m14.FSET_OF["sheikh_ada"]]; y = m14.M.B1_ms.values; t0 = time.time()
    model = m14.build("sheikh_ada", d["best"], d["rep"], 500).fit(X[tr], y[tr]); p = model.predict(X[te])
    diff = float(np.max(np.abs(p - d["pred_rb"]))); new_mae = float(np.abs(p - y[te]).mean()); old_mae = float(np.abs(d["pred_rb"] - y[te]).mean())
    return dict(key=key, kind=d["kind"], fold=d["fold"], config=d["best"], n_train=int(len(tr)), n_test=int(len(te)), seconds_new=round(time.time() - t0, 1), seconds_original=round(d["sec"], 1),
                mae_new=new_mae, mae_stored=old_mae, max_abs_pred_diff=diff, identical=bool(diff == 0.0))
if __name__ == "__main__":
    from joblib import Parallel, delayed
    t0 = time.time(); stamp = lambda: time.strftime("%H:%M:%S")
    print(f"[{stamp()}] re-training {len(KEYS)} AdaBoost models from scratch (500 trees each) with the recorded settings", flush=True)
    res = []
    for r in Parallel(n_jobs=len(KEYS), return_as="generator_unordered", backend="loky")(delayed(fit_one)(k) for k in KEYS):
        res.append(r); print(f"[{stamp()}] {r['key']}: config {r['config']} | trained on {r['n_train']} beats in {r['seconds_new']} s | MAE new {r['mae_new']:.4f} vs stored {r['mae_stored']:.4f} | max |prediction difference| {r['max_abs_pred_diff']} -> {'IDENTICAL' if r['identical'] else 'DIFFERENT'}", flush=True)
    res.sort(key=lambda r: r["key"]); ok = all(r["identical"] for r in res)
    m14 = load(); idx = np.where(m14.KEEP)[0]
    print(f"[{stamp()}] total {round(time.time() - t0)} s | all {len(res)} re-trained models give identical predictions: {ok}", flush=True)
    json.dump(dict(all_identical=ok, runs=res, finished=time.strftime("%Y-%m-%d %H:%M:%S")), open(ROOT / "results" / "reproduction_headline.json", "w"), indent=1)
