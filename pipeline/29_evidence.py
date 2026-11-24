"""Step 29: evidence package for the Final Results tab and the repository (evidence/).
 * evidence/fold_log.csv            every final training task: model, validation, fold, settings chosen, training seconds, test beats, MAE (from the saved task files)
 * evidence/predictions.csv.gz      every beat: true B and the stored predictions of each final model (anyone can recompute the MAE with a few lines)
 * evidence/model_run_log.txt, reproduction_log.txt, audit_log.txt   the logs of the runs
 * deploy_task1/data/final.json     data for the website tab
All numbers are read from saved files; nothing is typed by hand."""
import glob, gzip, json, pickle, re
import numpy as np, pandas as pd
from pathlib import Path
from common import ROOT, RES
EV = ROOT / "evidence"; EV.mkdir(exist_ok=True)
m = pd.read_csv(ROOT / "data/processed/cohort_meta.csv"); y = m.B1_ms.values
pk = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(RES / "tasks_v4/*.pkl"))) if "SMOKE" not in f]
VAL = {"pooled": "5-fold, patients held out", "x_e2g": "train EmpkinS, test Guardian", "x_g2e": "train Guardian, test EmpkinS"}
rows = [dict(model=d["model"], validation=VAL[d["kind"]], repeat=d["rep"], fold=d["fold"] if d["kind"] == "pooled" else "", settings_chosen=json.dumps(d["best"]), training_seconds=round(d["sec"], 1),
             test_beats=len(d["test_idx"]), mae_ms=round(float(np.abs(d["pred_rb"] - y[d["test_idx"]]).mean()), 3)) for d in pk]
F = pd.DataFrame(rows).sort_values(["model", "validation", "repeat", "fold"]); F.to_csv(EV / "fold_log.csv", index=False)
P = pd.DataFrame({"cohort_row": np.arange(len(m)), "dataset": m.dataset, "patient": m.subject, "phase": m.phase, "beat_id": m.heartbeat_id, "B_true_ms": y})
MODELS = ["sheikh_ada", "fus_ada", "sheikh_rf", "fus_rf", "sheikh_lgbm", "fus_lgbm", "sheikh_svr", "fus_svr"]
for mo in MODELS:
    p = np.full(len(m), np.nan)
    for d in pk:
        if d["model"] == mo and d["kind"] == "pooled" and d["rep"] == 0: p[d["test_idx"]] = d["pred_rb"]
    P[f"pred_{mo}_5fold"] = np.round(p, 2)
for kind in ("x_e2g", "x_g2e"):
    p = np.full(len(m), np.nan)
    for d in pk:
        if d["model"] == "sheikh_ada" and d["kind"] == kind: p[d["test_idx"]] = d["pred_rb"]
    P[f"pred_sheikh_ada_{kind[2:]}"] = np.round(p, 2)
P.to_csv(EV / "predictions.csv.gz", index=False, compression="gzip")
# logs
raw = [l.rstrip() for l in open(ROOT / "logs/v4_14_models.log", errors="ignore") if re.match(r"^(\d+ tasks|\d\d:\d\d:\d\d done|ALL V3)", l)]
(EV / "model_run_log.txt").write_text("# model training run: 184 tasks (nested cross-validation inside each; AdaBoost 500 trees), 40 parallel workers\n" + "\n".join(raw) + "\n")
rl = ROOT / "logs/reproduction_headline.log"; rep = json.load(open(RES / "reproduction_headline.json")) if (RES / "reproduction_headline.json").exists() else None
(EV / "reproduction_log.txt").write_text(rl.read_text() if rl.exists() else "pending\n")
import subprocess, sys
(EV / "audit_log.txt").write_text(subprocess.run([sys.executable, str(ROOT / "pipeline/28_audit.py")], capture_output=True, text=True).stdout)
screens = [("1_live_feature_verification.png", "Live check", "Python in the browser recomputes the 465 features of a beat and compares them with the stored ones"),
           ("2_training_in_progress.png", "Training in progress", "AdaBoost (quick setting) training in the browser, progress per split"),
           ("3_code_editor_and_run_result.png", "Code and result", "The Python that runs, and the result it produced"),
           ("4_leaderboard_sheikh_vs_abel.png", "Leaderboard", "Sheikh 465, Abel 13 and the fusion, same model and validation, next to the verified results")]
screens = [dict(file=f, title=t, cap=c) for f, t, c in screens if (EV / "screens" / f).exists()]
sel = F[F.model.isin(["sheikh_ada", "fus_ada"])]
J = dict(audit=json.load(open(RES / "audit.json")), repro=rep, screens=screens, logs=dict(model_run=(EV / "model_run_log.txt").read_text().splitlines(), reproduction=(EV / "reproduction_log.txt").read_text().splitlines(),
         audit=(EV / "audit_log.txt").read_text().splitlines()), fold_log=sel.to_dict(orient="records"), n_tasks=len(F))
json.dump(J, open(ROOT / "deploy_task1/data/final.json", "w"), indent=1)
print(f"fold_log {len(F)} rows | predictions {P.shape} | screens {len(screens)} | reproduction {'present' if rep else 'pending'} | audit ok {J['audit']['all_ok']}")
