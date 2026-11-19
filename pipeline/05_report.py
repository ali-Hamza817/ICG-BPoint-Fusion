"""Step 5: aggregate results/tasks/*.pkl -> tables (csv/md) + figures. Every number is computed here from saved predictions."""
import pickle, glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from common import *
sns.set_theme(style="whitegrid", font_scale=1.05); plt.rcParams.update({"savefig.dpi": 220, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
OUT = FIG / "results"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(PROC / "cohort_meta.csv"); y = m.B1_ms.values
files = [f for f in glob.glob(str(RES / "tasks/*.pkl")) if "SMOKE__" not in f]
D = [pickle.load(open(f, "rb")) for f in files]
print(len(D), "task files")

# ---- pooled subject-wise CV: pool all test folds of a repeat -> sample-wise MAE per repeat
rows = []; pooled = {}
for d in D:
    if d["kind"] != "pooled": continue
    pooled.setdefault((d["model"], d["fset"], d["target"], d["rep"]), []).append(d)
preds = {}
for k, ds in pooled.items():
    if len(ds) != 5: continue                                              # only complete repeats
    p = np.full(len(m), np.nan)
    for d in ds: p[d["test_idx"]] = d["pred_rb"]
    e = np.abs(p - y); preds[k] = p
    sub = pd.Series(e).groupby(m.subject.values).mean()
    rows.append(dict(model=k[0], fset=k[1], target=k[2], rep=k[3], MAE=e.mean(), SD=e.std(), MedAE=np.median(e), ME=(p - y).mean(), subj_MAE_sd=sub.std(), r=np.corrcoef(p, y)[0, 1], n=len(e)))
R = pd.DataFrame(rows)
if len(R):
    A = R.groupby(["model", "fset", "target"]).agg(MAE=("MAE", "mean"), MAE_rep_sd=("MAE", "std"), SD=("SD", "mean"), MedAE=("MedAE", "mean"), ME=("ME", "mean"), r=("r", "mean"), n_repeats=("rep", "count")).reset_index().sort_values("MAE")
    A.round(3).to_csv(RES / "table_pooled_cv.csv", index=False); open(RES / "table_pooled_cv.md", "w").write(A.round(2).to_markdown(index=False))
    print(A.round(2).to_string(index=False))

# ---- per-dataset breakdown of the best config (rep-averaged predictions)
if len(R):
    best = A.iloc[0]; ks = [k for k in preds if k[:3] == (best.model, best.fset, best.target)]
    pbar = np.mean([preds[k] for k in ks], axis=0); e = np.abs(pbar - y)
    per = pd.DataFrame({"MAE": pd.Series(e).groupby(m.dataset.values).mean(), "SD": pd.Series(e).groupby(m.dataset.values).std(), "n": pd.Series(e).groupby(m.dataset.values).size()})
    per.round(3).to_csv(RES / "table_best_by_dataset.csv"); print("best config", tuple(best[["model", "fset", "target"]]), "\n", per.round(2))
    # reference: inter-rater and rater2 evaluation
    b2 = m.B2_ms.values; ok = ~np.isnan(b2)
    ref = dict(inter_rater_MAE=float(np.abs(y[ok] - b2[ok]).mean()), model_vs_rater2_MAE=float(np.abs(pbar[ok] - b2[ok]).mean()), model_vs_rater1_MAE=float(e.mean()))
    json.dump(ref, open(RES / "reference_numbers.json", "w"), indent=1); print(ref)
    # error vs agreement level & HR
    ad = np.abs(y - b2); lvl = pd.cut(ad, [-1, 4, 10, 1e9], labels=["high ≤4", "med 4–10", "low >10"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.boxplot(x=lvl[ok], y=e[ok], ax=ax[0], showfliers=False); ax[0].set_title("Abs. error by inter-rater agreement"); ax[0].set_ylabel("|pred − rater1| (ms)")
    hg = pd.cut(m.HR, [0, 60, 100, 300], labels=["<60", "60–100", ">100"]); sns.boxplot(x=hg, y=e, ax=ax[1], showfliers=False); ax[1].set_title("Abs. error by heart-rate group"); ax[1].set_xlabel("bpm")
    ax[2].scatter(y, pbar, s=3, alpha=.2); lim = [-40, 230]; ax[2].plot(lim, lim, "k--"); ax[2].set_xlim(lim); ax[2].set_ylim(lim); ax[2].set_title("Predicted vs rater-1 B (RB, ms)"); ax[2].set_xlabel("rater 1"); ax[2].set_ylabel("predicted")
    fig.tight_layout(); fig.savefig(OUT / "best_model_diagnostics.png", bbox_inches="tight"); plt.close(fig)
    # model comparison bar chart
    top = A.assign(cfg=A.model + " | " + A.fset + " | " + A.target).head(20)
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top.cfg[::-1], top.MAE[::-1], xerr=top.MAE_rep_sd.fillna(0)[::-1], color="#2a6f97"); ax.set_xlabel("sample-wise MAE (ms), subject-wise 5-fold CV, mean of repeats ± SD")
    ax.set_title("Top 20 configurations"); fig.tight_layout(); fig.savefig(OUT / "model_ranking.png", bbox_inches="tight"); plt.close(fig)

# ---- cross-dataset
xr = []
for d in D:
    if d["kind"] == "pooled": continue
    te = d["test_idx"]; e = np.abs(d["pred_rb"] - y[te])
    xr.append(dict(direction="EmpkinS→Guardian" if d["kind"] == "x_emp2gdn" else "Guardian→EmpkinS", model=d["model"], fset=d["fset"], target=d["target"], MAE=e.mean(), SD=e.std(), ME=(d["pred_rb"] - y[te]).mean(), r=np.corrcoef(d["pred_rb"], y[te])[0, 1]))
if xr:
    X_ = pd.DataFrame(xr).sort_values(["direction", "MAE"]); X_.round(3).to_csv(RES / "table_cross_dataset.csv", index=False)
    print(X_.round(2).groupby("direction").head(8).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for a, (dr, g) in zip(ax, X_.groupby("direction")):
        g = g.assign(cfg=g.model + "|" + g.fset + "|" + g.target).head(15); a.barh(g.cfg[::-1], g.MAE[::-1], color="#c1440e"); a.set_title(f"Cross-dataset {dr}: top 15 by MAE"); a.set_xlabel("MAE (ms)")
    fig.tight_layout(); fig.savefig(OUT / "cross_dataset_ranking.png", bbox_inches="tight"); plt.close(fig)
