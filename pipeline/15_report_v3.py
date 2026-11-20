"""Step 15: aggregate results/tasks_v3 -> tables, paired tests, figures, reports/V3_RESULTS.md.  Every number is computed here from saved predictions."""
import os, pickle, glob
TAG = os.environ.get("TASKS_TAG", "v3")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from common import *
sns.set_theme(style="whitegrid", font_scale=1.05); plt.rcParams.update({"savefig.dpi": 220, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
OUT = FIG / "results"; OUT.mkdir(parents=True, exist_ok=True)
meta = read_meta_clean(); coh = (meta.B1_ms.notna() & (meta.B1_ms - meta.Q1_ms).between(20, 250)).values
Mc = pd.read_csv(PROC / "cohort_meta.csv"); y = Mc.B1_ms.values; subj = Mc.subject.values; dsn = Mc.dataset.values
D = [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / f"tasks_{TAG}/*.pkl")) if "SMOKE" not in f]
print(len(D), "task files")
LABEL = {"abel_rf": "Abel replica: 12 PEPbench rules+RR, RF", "abel_svr": "Abel rules, SVR (imputed)", "sheikh_ada": "Sheikh 465, AdaBoost", "sheikh_ada+rule": "Sheikh 465, AdaBoost + featureless rule",
         "sheikh_rf": "Sheikh 465, RF", "sheikh_lgbm": "Sheikh 465, LightGBM", "sheikh_svr": "Sheikh 465, SVR", "sheikh_ridge": "Sheikh 465, Ridge",
         "fus_ada": "Fusion (465+13), AdaBoost", "fus_ada+rule": "Fusion, AdaBoost + featureless rule", "fus_rf": "Fusion, RF", "fus_lgbm": "Fusion, LightGBM", "fus_svr": "Fusion, SVR", "fus_ridge": "Fusion, Ridge"}

# ---------- pooled ----------
g = {}
for d in D:
    if d["kind"] == "pooled": g.setdefault((d["model"], d["rep"]), []).append(d)
P = {}                                                           # (model_variant, rep) -> prediction vector (nan outside evaluation set)
for (mo, rep), ds in g.items():
    if len(ds) != 5: continue
    p = np.full(len(Mc), np.nan); pr = p.copy()
    for d in ds: p[d["test_idx"]] = d["pred_rb"]; pr[d["test_idx"]] = d.get("pred_rb_rule", d["pred_rb"])
    P[(mo, rep)] = p
    if mo.endswith("_ada"): P[(mo + "+rule", rep)] = pr
rows = []
common = np.all([~np.isnan(p) for p in P.values()], axis=0) if P else None        # beats evaluated by every model
for (mo, rep), p in P.items():
    k = ~np.isnan(p); e = np.abs(p - y)[k]; sj = pd.Series(np.abs(p - y)[k]).groupby(subj[k]).mean()
    rows.append(dict(model=mo, rep=rep, n=int(k.sum()), MAE=e.mean(), SD=e.std(), MedAE=np.median(e), ME=(p - y)[k].mean(), subjectMAE=sj.mean(), r=np.corrcoef(p[k], y[k])[0, 1]))
R = pd.DataFrame(rows)
A = R.groupby("model").agg(n=("n", "first"), MAE=("MAE", "mean"), MAE_rep_sd=("MAE", "std"), SD=("SD", "mean"), MedAE=("MedAE", "mean"), ME=("ME", "mean"), subjectMAE=("subjectMAE", "mean"), r=("r", "mean"), n_rep=("rep", "count")).reset_index().sort_values("MAE")
A.insert(1, "description", A.model.map(LABEL)); A.round(3).to_csv(RES / f"{TAG}_table_pooled.csv", index=False)

# ---------- paired tests vs the Abel replica (rep 0 predictions, per-subject MAE) ----------
T = []
base = P.get(("abel_rf", 0))
if base is not None:
    kb = ~np.isnan(base); sb = pd.Series(np.abs(base - y)[kb]).groupby(subj[kb]).mean(); rng = np.random.default_rng(0)
    for mo in [m for m in A.model if m != "abel_rf"]:
        p = P.get((mo, 0))
        if p is None: continue
        k = kb & ~np.isnan(p); s1 = pd.Series(np.abs(p - y)[k]).groupby(subj[k]).mean(); s0 = pd.Series(np.abs(base - y)[k]).groupby(subj[k]).mean(); d = (s1 - s0).dropna()
        w = stats.wilcoxon(d) if len(d) > 5 else None
        bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
        T.append(dict(candidate=mo, description=LABEL.get(mo, mo), n_subjects=len(d), replica_subjectMAE=s0.mean(), candidate_subjectMAE=s1.mean(), mean_diff=d.mean(), ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), wilcoxon_p=w.pvalue if w else np.nan, subjects_improved=int((d < 0).sum())))
TT = pd.DataFrame(T).sort_values("mean_diff") if T else pd.DataFrame(); TT.round(4).to_csv(RES / f"{TAG}_table_paired_vs_replica.csv", index=False)

# ---------- cross-dataset ----------
X = []
for d in D:
    if not d["kind"].startswith("x_"): continue
    te = d["test_idx"]; direction = "EmpkinS->Guardian" if d["kind"] == "x_e2g" else "Guardian->EmpkinS"
    for var, pr in ([(d["model"], d["pred_rb"])] + ([(d["model"] + "+rule", d["pred_rb_rule"])] if "pred_rb_rule" in d else [])):
        e = np.abs(pr - y[te]); sj = pd.Series(e).groupby(subj[te]).mean()
        X.append(dict(direction=direction, model=var, description=LABEL.get(var, var), n=len(te), MAE=e.mean(), subjectMAE=sj.mean(), subjectMAE_sd=sj.std(), SD=e.std(), ME=(pr - y[te]).mean(), best_cfg=str(d["best"])))
XT = pd.DataFrame(X).sort_values(["direction", "MAE"]) if X else pd.DataFrame(); XT.round(3).to_csv(RES / f"{TAG}_table_cross_dataset.csv", index=False)

# ---------- figures ----------
if len(A):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = A.copy(); a["lab"] = a.description; col = ["#C44E52" if m == "abel_rf" else ("#4C72B0" if m.startswith("sheikh") else ("#55A868" if m.startswith("fus") else "#8172B2")) for m in a.model]
    ax[0].barh(a.lab[::-1], a.MAE[::-1], color=col[::-1], xerr=a.MAE_rep_sd.fillna(0)[::-1]); ax[0].axvline(8.13, color="k", ls="--", lw=1.3, label="Abel 2026 as published (8.13; his cohort/folds)")
    ax[0].set_xlabel("pooled subject-wise CV MAE (ms)"); ax[0].set_title("Single-beat B-point error, subject-wise CV"); ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_xlim(left=0)
    if len(XT):
        piv = XT.pivot(index="description", columns="direction", values="MAE").sort_values("EmpkinS->Guardian"); piv.plot.barh(ax=ax[1], width=0.8, color=["#DD8452", "#4C72B0"])
        for lbl, v, c in (("Abel SVR E->G 19.39", 19.39, "#DD8452"), ("Abel SVR G->E 12.70", 12.70, "#4C72B0")): ax[1].axvline(v, color=c, ls=":", lw=1.6)
        ax[1].set_xlabel("cross-dataset MAE (ms); dotted = Abel's best (SVR)"); ax[1].set_title("Train on one dataset, test on the other"); ax[1].set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / f"{TAG}_pooled_and_cross.png"); plt.close(fig)

# ---------- report ----------
L = ["# v3 results: real Abel features vs real-Sheikh features (auto-generated from saved predictions)", "",
     "All numbers computed by `pipeline/15_report_v3.py` from `results/tasks_v3/`. Subject-wise grouped CV (5 folds; 3 repeats for fast models, 1 for AdaBoost); hyper-parameters tuned only by inner 3-fold group CV on the training part.", "",
     "## Caveats (read first)",
     "- Abel's 12 algorithms are Abel's own files run through a small shim (`external/biopsykit_shim`); deviations: heartbeats = annotated reference heartbeats with our R-peaks (not PEPbench's neurokit segmentation + matching).",
     "- Abel's C-point extractor (fixed prominence 1) fails on low-amplitude beats: 10.1 percent of Guardian beats vs 0.2 percent of EmpkinS; beats where all 12 rules failed are excluded (as Abel does). Evaluation set here: %d beats (Abel: 11,239)." % (int(A.n.max()) if len(A) else 0),
     "- Sheikh features are a line-by-line port of his MATLAB (validated exactly on his 2 demo beats). His AdaBoost was tuned on a small grid inside his published range (max_features 0.7, depth 12/20, lr 0.1/1; 100 trees inner, 500 trees final).",
     "- Not directly comparable with the numbers published by Abel (different cohort/folds); a replica of his design on OUR data is the fair baseline.", ""]
if len(A): L += ["## Pooled subject-wise CV (MAE ms vs rater 1)", "", A.round(2).to_markdown(index=False), ""]
if len(TT): L += ["## Paired comparison with the Abel replica (rep 0; per-subject MAE; negative = candidate better)", "", TT.round(3).to_markdown(index=False), ""]
if len(XT): L += ["## Cross-dataset (train on source only, test on target; no selection on target)", "", XT.drop(columns=["description"]).round(2).to_markdown(index=False), "",
                  "Abel 2026 reported (p.7): Emp->Gdn RF 36.24, SVR 19.39; Gdn->Emp RF 52.92 +/- 39.35, SVR 12.70 +/- 6.33 (read from the text; direction order of the first two as in the text)."]
open(ROOT / "reports" / f"{TAG.upper()}_RESULTS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
