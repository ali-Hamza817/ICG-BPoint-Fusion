"""Step 23: cross-dataset AdaBoost sensitivity. For both directions and both feature sets (Sheikh 465, fusion) train ALL four AdaBoost configurations
(lr 0.1/1.0 x depth 12/20, max_features 0.7, 500 trees, StandardScaler) on the source dataset and test on the target; no selection, no repeated tuning.
The nested inner-CV pick in step 14 changes between runs, so this table shows the full range.  Output: results/v4_ada_cross_all_configs.csv"""
import importlib.util, itertools, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
os.environ.setdefault("TASKS_TAG", "v4")
spec = importlib.util.spec_from_file_location("m14", Path(__file__).with_name("14_models_v3.py")); m14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m14)
M, FS, KEEP, FLESS, FB = m14.M, m14.FS, m14.KEEP, m14.FLESS, m14.FB_RB
idx = np.where(KEEP)[0]; ds = M.dataset.values; y = M.B1_ms.values
def run(direction, model, cfg):
    src, dst = ("EmpkinS", "Guardian") if direction == "EmpkinS->Guardian" else ("Guardian", "EmpkinS")
    tr, te = idx[ds[idx] == src], idx[ds[idx] == dst]; X = FS[m14.FSET_OF[model]]
    p = m14.build(model, cfg, 0, 500).fit(X[tr], y[tr]).predict(X[te]); pr = np.where(FLESS[te], FB[te], p)
    e, er = np.abs(p - y[te]), np.abs(pr - y[te])
    return dict(direction=direction, model=model, lr=cfg["lr"], depth=cfg["depth"], n_test=len(te), MAE=e.mean(), MAE_with_featureless_rule=er.mean(), ME=(p - y[te]).mean(), SD=e.std())
jobs = [(d, mo, dict(lr=lr, depth=dp)) for d in ("EmpkinS->Guardian", "Guardian->EmpkinS") for mo in ("sheikh_ada", "fus_ada") for lr, dp in itertools.product((0.1, 1.0), (12, 20))]
rows = Parallel(n_jobs=16)(delayed(run)(*j) for j in jobs)
df = pd.DataFrame(rows); df.round(3).to_csv(m14.RES / "v4_ada_cross_all_configs.csv", index=False); print(df.round(2).to_string(index=False))
