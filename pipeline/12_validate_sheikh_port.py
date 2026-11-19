"""Step 12: validate the Python port (03c) against Sheikh's own MATLAB output on his two demo beats.
Input  : external/sheikh_toolbox_B_point/Demo Data/Demo_102D3_B_{1,2}.mat   (his ICG, D1, D2, D2_sign, RC, vel_index, RB_out)
Truth  : external/sheikh_toolbox_B_point/Demo Data/features_final_demo_465.xlsx (his MATLAB output feature vectors)
Result : results/sheikh_port_validation.json  (max abs deviation per block; exact-equality flags)"""
import json, importlib.util
import numpy as np, pandas as pd, scipy.io as sio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DEMO = ROOT / "external/sheikh_toolbox_B_point/Demo Data"
spec = importlib.util.spec_from_file_location("f03c", Path(__file__).with_name("03c_sheikh_features_v2.py")); f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
truth = pd.read_excel(DEMO / "features_final_demo_465.xlsx"); T = truth.iloc[:, 2:467].values.astype(float)   # 465 features, then RB_out
rep = []
for k in (1, 2):
    r = sio.loadmat(DEMO / f"Demo_102D3_B_{k}.mat", squeeze_me=True, struct_as_record=False)["Rec_feat"]
    mag_B = r.feat_icg[int(r.RB_out)]; mag_C = r.vel_index + mag_B                                       # exactly as in his MATLAB (line 70-71)
    v, aux = f.sheikh_core(r.feat_icg, r.feat_icg_D1, r.feat_icg_D2, r.feat_icg_D2_sign, float(r.RC), float(mag_C))
    t = T[k - 1]; hl = slice(450, 465)
    rep.append(dict(beat=k, n_features=len(v), max_abs_dev_timeseries_450=float(np.abs(v[:450] - t[:450]).max()),
                    max_abs_dev_highlevel_15=float(np.abs(v[hl] - t[hl]).max()),
                    highlevel_exact_match=bool(np.allclose(v[hl], t[hl], atol=1e-9)),
                    ours_highlevel=dict(zip(f.HL_NAMES, np.round(v[hl], 4).tolist())), his_highlevel=dict(zip(f.HL_NAMES, np.round(t[hl], 4).tolist()))))
(ROOT / "results").mkdir(exist_ok=True); json.dump(rep, open(ROOT / "results/sheikh_port_validation.json", "w"), indent=1)
for x in rep: print(x["beat"], "n_features", x["n_features"], "| max dev timeseries", x["max_abs_dev_timeseries_450"], "| max dev high-level", x["max_abs_dev_highlevel_15"], "| exact:", x["highlevel_exact_match"])
