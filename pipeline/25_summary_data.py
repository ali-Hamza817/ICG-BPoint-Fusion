"""Step 25: numbers for the website's Summary tab, generated from the result files (never typed by hand).
Inputs: results/final_pooled.csv, v4_ada_cross_all_configs.csv, v4_table_cross_dataset.csv, ea_vs_single_v4_w{10,20}.csv, artefact_summary.json, dataset_task2/task2_beats.csv.
Output: deploy_task1/data/summary.json (encrypted into the site by step 21)."""
import json
import numpy as np, pandas as pd
from common import ROOT, RES
t = pd.read_csv(ROOT / "dataset_task2" / "task2_beats.csv"); u = t[t.usable == 1]
art = json.load(open(RES / "artefact_summary.json")); P = pd.read_csv(RES / "final_pooled.csv"); C = pd.read_csv(RES / "v4_ada_cross_all_configs.csv"); X = pd.read_csv(RES / "v4_table_cross_dataset.csv")
b = u[u.B_ms_after_R_rater2.notna()]; d = (b.B_ms_after_R_rater1 - b.B_ms_after_R_rater2).abs()
S = dict(
    counts=dict(patients=int(t.patient.nunique()), annotated=int(len(t)), usable=int(len(u)), empkins=int((u.dataset == "EmpkinS").sum()), guardian=int((u.dataset == "Guardian").sum()),
                evaluated=int(P.n_beats.max()), removed_artefact=int(art["removed_from_usable"]), matlab_ok=int(u.matlab_ok.sum()), matlab_not=int(len(u) - u.matlab_ok.sum())),
    data={ds: dict(RB=float(g.B_ms_after_R_rater1.mean()), PEP=float(g.PEP_ms_rater1.mean()), HR=float(g.HR_bpm.mean())) for ds, g in u.groupby("dataset")},
    inter_rater=dict(mean=float(d.mean()), sd=float(d.std()), n=int(len(d))),
    pooled=[dict(model=r.model, desc=r.description, MAE=float(r.MAE), SD=float(r.SD), MedAE=float(r.MedAE), lo=float(r.subject_MAE_ci95_lo), hi=float(r.subject_MAE_ci95_hi), reps=int(r.n_repeats)) for r in P.itertuples()],
    cross={}, ea={}, published=dict(pooled=8.13, pooled_sd=12.12, inter_rater=8.35, inter_rater_sd=14.78, e2g_RF=36.24, e2g_SVR=19.39, g2e_RF=52.92, g2e_SVR=12.70))
for dr in ("EmpkinS->Guardian", "Guardian->EmpkinS"):
    g = C[C.direction == dr]; pick = X[(X.direction == dr) & X.model.isin(["sheikh_ada", "fus_ada"])]
    S["cross"][dr] = dict(min=float(g.MAE.min()), max=float(g.MAE.max()), best_pick=float(pick.MAE.min()), n_settings=int(len(g)))
for w in (10, 20):
    e = pd.read_csv(RES / f"ea_vs_single_v4_w{w}.csv"); a = e[e.approach.str.startswith("EA model")]
    S["ea"][str(w)] = dict(n_windows=int(a.n_windows.max()), ea_best=float(a.MAE.min()), ea_ada=float(a[a.approach == "EA model: sheikh_ada"].MAE.iloc[0]),
                           single_avg=float(e[e.approach == "single->avg:sheikh_ada"].MAE.iloc[0]), single_beat=float(e[e.approach.str.contains("single-beat error") & e.approach.str.endswith("sheikh_ada")].MAE.iloc[0]))
json.dump(S, open(ROOT / "deploy_task1" / "data" / "summary.json", "w"), indent=1); print(json.dumps(S, indent=1)[:1800])
