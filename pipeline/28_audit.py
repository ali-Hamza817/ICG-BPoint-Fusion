"""Step 28: independent audit of the reported numbers.
(1) recompute every pooled MAE from the raw saved predictions (results/tasks_v4/*.pkl) with a few lines of numpy, WITHOUT the report scripts, and compare with the reported tables;
(2) recompute the cross-dataset numbers the same way; (3) check the numbers quoted from Abel et al. 2026 against the text of his PDF.
Output: results/audit.json (and a printed log)."""
import glob, json, pickle, re, subprocess, sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
m = pd.read_csv(ROOT / "data/processed/cohort_meta.csv"); y = m.B1_ms.values
pk = [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / "tasks_v4/*.pkl")) if "SMOKE" not in f]
out = dict(pooled=[], cross=[], published=[], notes=[]); worst = 0.0
rep = pd.read_csv(RES / "final_pooled.csv").set_index("model")
for model in sorted({d["model"] for d in pk if d["kind"] == "pooled" and d["model"] in rep.index}):
    maes = []
    for r in sorted({d["rep"] for d in pk if d["kind"] == "pooled" and d["model"] == model}):
        ds = [d for d in pk if d["kind"] == "pooled" and d["model"] == model and d["rep"] == r]
        if len(ds) != 5: continue
        p = np.full(len(m), np.nan)
        for d in ds: p[d["test_idx"]] = d["pred_rb"]
        ok = ~np.isnan(p); maes.append(float(np.abs(p[ok] - y[ok]).mean())); n = int(ok.sum())
    mine = float(np.mean(maes)); rptd = float(rep.loc[model, "MAE"]); worst = max(worst, abs(mine - rptd))
    out["pooled"].append(dict(model=model, repeats=len(maes), n_beats=n, mae_recomputed=round(mine, 4), mae_reported=round(rptd, 4), abs_diff=round(abs(mine - rptd), 6)))
    print(f"pooled {model:12s} recomputed {mine:.4f} | reported {rptd:.4f} | diff {abs(mine - rptd):.1e}")
cr = pd.read_csv(RES / "v4_ada_cross_all_configs.csv")
for dname, dr in (("e2g", "EmpkinS->Guardian"), ("g2e", "Guardian->EmpkinS")):
    for model in ("sheikh_ada", "fus_ada"):
        d = [x for x in pk if x["kind"] == f"x_{dname}" and x["model"] == model][0]; te = d["test_idx"]; mine = float(np.abs(d["pred_rb"] - y[te]).mean())
        rows = cr[(cr.direction == dr) & (cr.model == model)]; inrange = bool(rows.MAE.min() - 1e-3 <= mine <= rows.MAE.max() + 1e-3)
        out["cross"].append(dict(direction=dr, model=model, mae_selected_run=round(mine, 4), range_min=round(float(rows.MAE.min()), 4), range_max=round(float(rows.MAE.max()), 4), inside_reported_range=inrange))
        print(f"cross {dr:18s} {model:11s} stored selected-run MAE {mine:.3f} | all-settings range {rows.MAE.min():.3f}-{rows.MAE.max():.3f} | inside range: {inrange}")
txt = subprocess.run(["pdftotext", "-layout", str(next(ROOT.glob("2026_Abel*.pdf"))), "-"], capture_output=True, text=True).stdout; flat = re.sub(r"\s+", " ", txt)
for label, pat in (("pooled MAE 8.13 +- 12.12", r"8\.13 \D{1,3}12\.12"), ("inter-rater 8.35 +- 14.78", r"8\.35 \D{1,3}14\.78"), ("cross-dataset RF 36.24", r"36\.24 ms"), ("cross-dataset SVR 19.39", r"19\.39 ms"),
                   ("reverse RF 52.92 +- 39.35", r"52\.92 \D{1,3}39\.35"), ("reverse SVR 12.70 +- 6.33", r"12\.70 \D{1,3}6\.33"), ("best traditional Drost 17.9 +- 23.0", r"17\.9 \D{1,3}23\.0")):
    found = bool(re.search(pat, flat)); out["published"].append(dict(quote=label, found_in_pdf=found)); print(f"Abel paper contains '{label}': {found}")
out["largest_pooled_difference"] = worst; out["all_ok"] = worst < 1e-3 and all(c["inside_reported_range"] for c in out["cross"]) and all(p["found_in_pdf"] for p in out["published"])
json.dump(out, open(RES / "audit.json", "w"), indent=1); print("AUDIT OK:", out["all_ok"], "| largest pooled difference", f"{worst:.1e}")
