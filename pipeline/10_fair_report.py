"""Step 10: aggregate results/tasks_fair (+ step-4 Sheikh-feature pooled/AdaBoost results) -> fair tables, tests, figure, report.
Every number is computed here from saved predictions. Selection of the cross-dataset candidate uses SOURCE-only CV."""
import pickle, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from common import *
sns.set_theme(style="whitegrid", font_scale=1.05); plt.rcParams.update({"savefig.dpi": 220, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
m = pd.read_csv(PROC / "cohort_meta.csv"); y = m.B1_ms.values; subj = m.subject.values; ds_of = m.dataset.values
load = lambda pat: [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / pat)) if "SMOKE" not in f]
D = load("tasks_fair/*.pkl"); D4 = load("tasks/*.pkl")
print(len(D), "fair task files;", len(D4), "step-4 task files")
REFS = {"Abel RF  Emp->Gdn": 36.24, "Abel SVR Emp->Gdn": 19.39, "Abel RF  Gdn->Emp": 52.92, "Abel SVR Gdn->Emp": 12.70}   # Abel 2026 p.7 (verified in PDF)

# ---------- pooled CV (pool test folds of each repeat) ----------
def pooled_preds(Dl):
    g = {}
    for d in Dl:
        if d["kind"] == "pooled": g.setdefault((d["model"], d["fset"], d["target"], d["rep"]), []).append(d)
    P = {}
    for k, ds in g.items():
        if len(ds) != 5: continue
        p = np.full(len(m), np.nan)
        for d in ds: p[d["test_idx"]] = d["pred_rb"]
        P[k] = p
    return P
P = {**pooled_preds(D4), **pooled_preds(D)}
rows = []
for k, p in P.items():
    e = np.abs(p - y); sm = pd.Series(e).groupby(subj).mean()
    rows.append(dict(model=k[0], fset=k[1], target=k[2], rep=k[3], MAE=e.mean(), SD=e.std(), MedAE=np.median(e), ME=(p - y).mean(), subjectMAE=sm.mean(), r=np.corrcoef(p, y)[0, 1]))
R = pd.DataFrame(rows)
A = R.groupby(["model", "fset", "target"]).agg(MAE=("MAE", "mean"), MAE_rep_sd=("MAE", "std"), SD=("SD", "mean"), MedAE=("MedAE", "mean"), ME=("ME", "mean"), subjectMAE=("subjectMAE", "mean"), r=("r", "mean"), n_rep=("rep", "count")).reset_index().sort_values("MAE")
A.round(3).to_csv(RES / "fair_table_pooled.csv", index=False)
REPLICA = ("rf", "abel11nan", "RB")
def rep_pred(key): return np.mean([p for k, p in P.items() if k[:3] == key], axis=0)
def subj_mae(p): return pd.Series(np.abs(p - y)).groupby(subj).mean()

# ---------- paired tests vs the Abel replica (per-subject MAE, Wilcoxon signed-rank + bootstrap CI of mean difference) ----------
tests = []
if any(k[:3] == REPLICA for k in P):
    base = subj_mae(rep_pred(REPLICA))
    names = {"A-priori 1: Sheikh465 + RF (RB)": ("rf", "sheikh465", "RB"), "A-priori 2: Fusion(Sheikh465+Abel11) + RF (RB)": ("rf", "fusion_a", "RB"),
             "A-priori 3: Fusion + per-subject z + RF (RB)": ("rf", "fusion_a_sz", "RB")}
    cand = A[~((A.model == "rf") & (A.fset == "abel11nan"))]
    if len(cand): b = cand.iloc[0]; names[f"Selected-by-pooled-MAE (optimistic): {b.model}/{b.fset}/{b.target}"] = (b.model, b.fset, b.target)
    rng = np.random.default_rng(0)
    for lab, key in names.items():
        if not any(k[:3] == key for k in P): continue
        s = subj_mae(rep_pred(key)); d = (s - base).dropna(); w = stats.wilcoxon(d) if len(d) > 5 else None
        bs = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
        tests.append(dict(comparison=lab, replica_subjectMAE=base.mean(), candidate_subjectMAE=s.mean(), mean_diff=d.mean(), ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5),
                          wilcoxon_p=w.pvalue if w else np.nan, n_subjects=len(d), subjects_improved=int((d < 0).sum())))
T = pd.DataFrame(tests); T.round(4).to_csv(RES / "fair_table_paired_tests.csv", index=False)

# ---------- cross-dataset with SOURCE-only selection ----------
def cross_tables():
    out = []
    for dirn, lab, src, dst in (("e2g", "EmpkinS->Guardian", "EmpkinS", "Guardian"), ("g2e", "Guardian->EmpkinS", "Guardian", "EmpkinS")):
        cv, xs = {}, {}
        for d in D:
            key = (d["model"], d["fset"], d["target"])
            if d["kind"] == f"srccv_{dirn}": cv.setdefault(key, {})[d["fold"]] = d
            if d["kind"] == f"x_{dirn}": xs[key] = d
        for key, folds in cv.items():
            if len(folds) != 5 or key not in xs: continue
            idx = np.concatenate([f["test_idx"] for f in folds.values()]); pr = np.concatenate([f["pred_rb"] for f in folds.values()])
            src_mae = np.abs(pr - y[idx]).mean(); x = xs[key]; e = np.abs(x["pred_rb"] - y[x["test_idx"]])
            out.append(dict(direction=lab, model=key[0], fset=key[1], target=key[2], sourceCV_MAE=src_mae, target_MAE=e.mean(), target_subjectMAE=pd.Series(e).groupby(subj[x["test_idx"]]).mean().mean(),
                            target_SD=e.std(), target_ME=(x["pred_rb"] - y[x["test_idx"]]).mean()))
    return pd.DataFrame(out)
X = cross_tables()
if len(X):
    X.round(3).to_csv(RES / "fair_table_cross_all_candidates.csv", index=False)
    sel = []
    for lab, g in X.groupby("direction"):
        pool = g[g.fset != "abel11nan"]; s = pool.sort_values("sourceCV_MAE").iloc[0].copy(); s["rule"] = "selected by SOURCE-only CV (48 candidates)"; sel.append(s)
        for nm, q in (("Abel-style RF (replica, NaN kept)", g[(g.model == "rf") & (g.fset == "abel11nan")]), ("Abel-style SVR on rule feats (abel11)", g[(g.model == "svr") & (g.fset == "abel11") & (g.target == "RB")]),
                      ("Sheikh feats + SVR (sheikh465)", g[(g.model == "svr") & (g.fset == "sheikh465") & (g.target == "RB")]), ("Sheikh feats + RF (sheikh465)", g[(g.model == "rf") & (g.fset == "sheikh465") & (g.target == "RB")])):
            if len(q): r = q.iloc[0].copy(); r["rule"] = "fixed a-priori: " + nm; sel.append(r)
        t = g.sort_values("target_MAE").iloc[0].copy(); t["rule"] = "ORACLE best-on-target (NOT a valid claim; shown only for transparency)"; sel.append(t)
    S = pd.DataFrame(sel); S.round(3).to_csv(RES / "fair_table_cross_selected.csv", index=False)

# ---------- figure ----------
if len(A):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sub = A[A.target == "RB"].copy(); sub["cfg"] = sub.model + " | " + sub.fset
    sub = sub.sort_values("MAE").head(14); ax[0].barh(sub.cfg[::-1], sub.MAE[::-1], xerr=sub.MAE_rep_sd.fillna(0)[::-1], color="#4C72B0")
    ax[0].axvline(8.13, color="crimson", ls="--", lw=1.4, label="Abel 2026 reported (8.13)"); ax[0].set_xlabel("pooled subject-wise CV MAE (ms)"); ax[0].set_title("Pooled CV: top configs (target = RB)"); ax[0].legend(); ax[0].set_xlim(left=0)
    if len(X):
        for i, (lab, g) in enumerate(X.groupby("direction")):
            ax[1].scatter(g.sourceCV_MAE, g.target_MAE, s=25, alpha=.6, label=lab)
        ax[1].set_xlabel("source-only CV MAE (ms)"); ax[1].set_ylabel("target-dataset MAE (ms)"); ax[1].set_title("Cross-dataset: every candidate (selection uses x-axis only)"); ax[1].legend()
    fig.tight_layout(); (FIG / "results").mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "results" / "fair_pooled_and_cross.png"); plt.close(fig)

# ---------- report ----------
L = ["# Fair-evaluation results (auto-generated from saved predictions)", "",
     "All numbers computed by `pipeline/10_fair_report.py` from `results/tasks_fair/` and `results/tasks/`. Subject-wise grouped CV, 5 folds x 3 repeats; hyper-parameters tuned by inner 3-fold group-CV on training data only.", "",
     "## Caveats (read first)",
     "- Abel features are a **re-implementation** (11 of 12 rules, `pipeline/08_abel_features.py`); PEPbench/biopsykit could not be installed (PyPI ~2 KB/s). Miljkovic 2022 omitted. Replica numbers are therefore not guaranteed to equal Abel's.",
     "- Cohort here: 11,903 beats vs Abel's 11,239 (different exclusion rules); folds differ from Abel's. Reference numbers from the paper are context, not a like-for-like comparison.",
     "- 'Selected-by-pooled-MAE' picks the best of many configs on the same CV folds, hence optimistic; the a-priori comparisons are the defensible ones.",
     "- Cross-dataset: candidate chosen by source-only CV. The ORACLE row (best on target) is shown for transparency only.", ""]
if len(A): L += ["## Pooled subject-wise CV (all target = RB unless 'dRC' = predict B-C offset; MAE ms, mean over 3 repeats)", "", A.head(25).round(2).to_markdown(index=False), "", f"Abel 2026 reported: 8.13 +/- 12.12 ms (sample-wise, his cohort/folds)."]
if len(T): L += ["", "## Paired tests vs Abel replica (per-subject MAE; negative diff = candidate better)", "", T.round(3).to_markdown(index=False)]
if len(X): L += ["", "## Cross-dataset (train on source only, test on target)", "", S[["direction", "rule", "model", "fset", "target", "sourceCV_MAE", "target_MAE", "target_subjectMAE", "target_SD"]].round(2).to_markdown(index=False),
                 "", "Abel 2026 reported (p.7): Emp->Gdn RF 36.24, SVR 19.39 (sample-wise); Gdn->Emp RF 52.92 +/- 39.35, SVR 12.70 +/- 6.33 (subject-wise). Direction order of the first two is as read from the text."]
open(ROOT / "reports" / "FAIR_RESULTS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
