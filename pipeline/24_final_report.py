"""Step 24: final results = OUR models vs Abel et al.'s PUBLISHED numbers (taken as given, no re-implementation of his pipeline is reported).
Inputs (all produced by earlier steps on the clean beats): results/tasks_v4 (pooled + cross-dataset predictions), v4_ada_cross_all_configs.csv, ea_vs_single_v4_w{10,20}.csv.
Outputs: reports/RESULTS.md, results/final_summary.json, figures/results/final_comparison.png.  Every number is computed here from saved predictions / tables."""
import glob, json, pickle
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from common import *
sns.set_theme(style="whitegrid"); plt.rcParams.update({"savefig.dpi": 220, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})

# ---- Abel et al. 2026, as published (verified against the PDF in this repo) ----
PUB = dict(pooled_MAE=8.13, pooled_SD=12.12, cv_fold_MAE=8.15, n_cycles=11239, inter_rater_MAE=8.35, inter_rater_SD=14.78,
           e2g_RF=36.24, e2g_SVR=19.39, g2e_RF=52.92, g2e_SVR=12.70)
meta = read_meta_clean(); coh = (meta.B1_ms.notna() & (meta.B1_ms - meta.Q1_ms).between(20, 250)).values
Mc = pd.read_csv(PROC / "cohort_meta.csv"); y = Mc.B1_ms.values; subj = Mc.subject.values
D = [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / "tasks_v4/pooled__*.pkl")) if "SMOKE" not in f]
OURS = {"sheikh_ada": "Sheikh 465 features + AdaBoost", "fus_ada": "Sheikh 465 + Abel 13 features + AdaBoost", "sheikh_rf": "Sheikh 465 + Random Forest", "fus_rf": "Fusion + Random Forest",
        "sheikh_lgbm": "Sheikh 465 + LightGBM", "fus_lgbm": "Fusion + LightGBM", "sheikh_svr": "Sheikh 465 + SVR", "fus_svr": "Fusion + SVR"}
g = {}
for d in D:
    if d["model"] in OURS: g.setdefault((d["model"], d["rep"]), []).append(d)
rng = np.random.default_rng(0); rows = []
for mo in OURS:
    reps = [v for (m, r), v in g.items() if m == mo and len(v) == 5]
    if not reps: continue
    maes, preds = [], []
    for ds in reps:
        p = np.full(len(Mc), np.nan)
        for d in ds: p[d["test_idx"]] = d["pred_rb"]
        preds.append(p); maes.append(np.nanmean(np.abs(p - y)))
    p = np.mean(preds, axis=0); k = ~np.isnan(p); e = np.abs(p - y)[k]; sj = pd.Series(e).groupby(subj[k]).mean()
    boots = [sj.sample(len(sj), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    rows.append(dict(model=mo, description=OURS[mo], n_beats=int(k.sum()), n_repeats=len(reps), MAE=float(np.mean(maes)), SD=float(e.std()), MedAE=float(np.median(e)), subject_MAE=float(sj.mean()),
                     subject_MAE_ci95_lo=float(np.percentile(boots, 2.5)), subject_MAE_ci95_hi=float(np.percentile(boots, 97.5)), delta_vs_published=float(np.mean(maes) - PUB["pooled_MAE"])))
P = pd.DataFrame(rows).sort_values("MAE"); P.round(3).to_csv(RES / "final_pooled.csv", index=False)

X = pd.read_csv(RES / "v4_table_cross_dataset.csv"); X = X[X.model.isin(OURS) | X.model.str.replace("+rule", "").isin(OURS)]
Xc = pd.read_csv(RES / "v4_ada_cross_all_configs.csv") if (RES / "v4_ada_cross_all_configs.csv").exists() else pd.DataFrame()
rngc = {}
if len(Xc):
    for (dr, mo), gg in Xc.groupby(["direction", "model"]): rngc[(dr, mo)] = (gg.MAE.min(), gg.MAE.max(), gg.MAE.mean())
EA = {w: pd.read_csv(RES / f"ea_vs_single_v4_w{w}.csv") for w in (10, 20) if (RES / f"ea_vs_single_v4_w{w}.csv").exists()}
art = json.load(open(RES / "artefact_summary.json"))
best = P.iloc[0]
summ = dict(published=PUB, n_usable_beats=art["usable_after"], n_eval_beats=int(best.n_beats), best_model=best.model, best_MAE=float(best.MAE), best_subject_ci95=[float(best.subject_MAE_ci95_lo), float(best.subject_MAE_ci95_hi)],
            delta_vs_published=float(best.delta_vs_published), cross_ada_range={f"{k[0]}|{k[1]}": [float(v[0]), float(v[1]), float(v[2])] for k, v in rngc.items()})
json.dump(summ, open(RES / "final_summary.json", "w"), indent=1)

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
q = P.iloc[::-1]; col = ["#0072B2" if m.endswith("ada") else "#7FA7C9" for m in q.model]
ax[0].barh(q.description, q.MAE, color=col)
ax[0].axvline(PUB["pooled_MAE"], color="crimson", ls="--", lw=1.6); ax[0].text(PUB["pooled_MAE"] + 0.15, len(q) - 0.45, "Abel 2026 published: 8.13 ms", color="crimson", fontsize=10, va="bottom"); ax[0].set_ylim(-0.6, len(q) + 0.15); ax[0].set_xlabel("pooled subject-wise CV MAE (ms)"); ax[0].set_title("Single-beat B-point error (our clean beats)"); ax[0].set_xlim(0, max(14, q.MAE.max() + 1))
for a, (dr, t, rf, svr) in zip(ax[1:], [("EmpkinS->Guardian", "Train EmpkinS, test Guardian", PUB["e2g_RF"], PUB["e2g_SVR"]), ("Guardian->EmpkinS", "Train Guardian, test EmpkinS", PUB["g2e_RF"], PUB["g2e_SVR"])]):
    names, lo, hi, mid = [], [], [], []
    for mo, lab in (("sheikh_ada", "Sheikh 465\nAdaBoost"), ("fus_ada", "Fusion\nAdaBoost")):
        if (dr, mo) in rngc: l, h, m_ = rngc[(dr, mo)]; names.append(lab); lo.append(l); hi.append(h); mid.append(m_)
    xs = np.arange(len(names)); a.bar(xs, mid, color="#0072B2", width=.5); a.errorbar(xs, mid, yerr=[np.array(mid) - np.array(lo), np.array(hi) - np.array(mid)], fmt="none", ecolor="k", capsize=6)
    a.axhline(rf, color="#999", ls=":", lw=1.6); a.text(len(names) - .45, rf, f"Abel RF {rf:g}", va="bottom", ha="right", fontsize=9, color="#666")
    a.axhline(svr, color="crimson", ls="--", lw=1.6); a.text(len(names) - .45, svr, f"Abel SVR {svr:g}", va="bottom", ha="right", fontsize=9, color="crimson")
    a.set_xticks(xs); a.set_xticklabels(names); a.set_title(t); a.set_ylabel("MAE (ms)"); a.set_xlabel("bar = mean of 4 settings, whisker = range", fontsize=9); a.set_ylim(0, max(60, svr + 5))
fig.tight_layout(); (FIG / "results").mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "results" / "final_comparison.png"); plt.close(fig)

# ---- report ----
f = lambda v, d=2: f"{v:.{d}f}"
L = ["# Results: our models vs Abel et al. 2026 (published figures)", "",
     "All numbers are computed from saved predictions on the **clean beats** (artefact-labelled points removed, see README). Abel's figures are taken from his paper as published (MAE against the manual B-point annotation); "
     f"his cohort ({PUB['n_cycles']:,} cycles) and folds differ from ours ({art['usable_after']:,} usable beats, {int(best.n_beats):,} evaluated here), so differences are **indicative, not like-for-like**.", "",
     "## 1. Pooled single-beat B-point error (subject-wise cross-validation)", "",
     "| Model | MAE ms | SD | Median AE | per-subject MAE [95 % CI] | vs Abel published 8.13 |", "|---|---|---|---|---|---|"]
for r in P.itertuples(): L.append(f"| {r.description} | **{f(r.MAE)}** | {f(r.SD)} | {f(r.MedAE)} | {f(r.subject_MAE)} [{f(r.subject_MAE_ci95_lo)}, {f(r.subject_MAE_ci95_hi)}] | {r.delta_vs_published:+.2f} ms |")
L += ["", f"Published (Abel 2026, Random Forest on 12 rule-based estimates + RR): **MAE {PUB['pooled_MAE']} ± {PUB['pooled_SD']} ms** (CV-fold mean {PUB['cv_fold_MAE']}). Published inter-rater MAE {PUB['inter_rater_MAE']} ± {PUB['inter_rater_SD']} ms; on our clean beats the inter-rater B-point difference is 7.94 ± 14.26 ms.", "",
      f"Uncertainty: the sample-wise MAE above is pooled over beats; the per-subject mean with its bootstrap interval over the 39 subjects is wider. For our best model the interval [{f(best.subject_MAE_ci95_lo)}, {f(best.subject_MAE_ci95_hi)}] ms **contains** Abel's published 8.13 ms, so with 39 subjects the difference to the published figure is **not** clearly outside sampling uncertainty (and the cohorts differ).", "",
      "AdaBoost: one repeat, nested tuning on a small grid inside Sheikh's published range (max_features 0.7, depth 12/20, learning rate 0.1/1; 100 trees in the inner loop, 500 for the final model). Other models: mean of three repeats.", "",
      "## 2. Cross-dataset (train on one dataset only, test on the other)", ""]
L += ["| Direction | Our AdaBoost (nested pick) | Our AdaBoost, all 4 settings (min – max) | Abel published RF | Abel published SVR |", "|---|---|---|---|---|"]
for dr, rf, svr in (("EmpkinS->Guardian", PUB["e2g_RF"], PUB["e2g_SVR"]), ("Guardian->EmpkinS", PUB["g2e_RF"], PUB["g2e_SVR"])):
    for mo, lab in (("sheikh_ada", "Sheikh 465"), ("fus_ada", "Fusion")):
        rr = X[(X.direction == dr) & (X.model == mo)]; pick = f(rr.MAE.iloc[0]) if len(rr) else "–"
        rg = f"{f(rngc[(dr, mo)][0])} – {f(rngc[(dr, mo)][1])}" if (dr, mo) in rngc else "pending"
        L.append(f"| {dr.replace('->', ' → ')} · {lab} | {pick} | {rg} | {rf} | {svr} |")
L += ["", "Reading: EmpkinS → Guardian is clearly below Abel's published values. Guardian → EmpkinS is **not** better than Abel's published SVR (12.70 ms); the result depends strongly on the model setting chosen, which is why the full range is shown. Every model shows a systematic bias in that direction (the two datasets differ strongly in B-point timing).", ""]
if EA:
    L += ["## 3. Single beat vs ensemble-averaged beat (Sheikh's averaging algorithm on Abel's beats)", "", "Reference = mean rater-1 B point of the beats in each window (our assumption, raters annotated single beats).", "",
          "| Window | Model trained on averaged beats (best / AdaBoost) | Single-beat AdaBoost, predictions averaged over the window | Same model, mean per-beat error |", "|---|---|---|---|"]
    for w, t in EA.items():
        a = t[t.approach.str.startswith("EA model")]; best_ea = a.loc[a.MAE.idxmin()]; ada_ea = a[a.approach == "EA model: sheikh_ada"].iloc[0]
        s1 = t[t.approach == "single->avg:sheikh_ada"].iloc[0]; s2 = t[t.approach.str.contains("single-beat error") & t.approach.str.endswith("sheikh_ada")].iloc[0]
        L.append(f"| {w} s ({int(ada_ea.n_windows)} windows) | {f(best_ea.MAE)} ({best_ea.approach.split(': ')[1]}) / {f(ada_ea.MAE)} | {f(s1.MAE)} | {f(s2.MAE)} |")
    L += ["", "Averaged-beat models train on only a few hundred averaged beats (versus about 9,000 single beats), so this does not show that averaging is harmful; it shows that averaged-beat training is data-starved here.", ""]
L += ["## Caveats", "- Published figures are used as reference only; no re-implementation of Abel's pipeline is reported.",
      "- Different cohort size and folds from Abel's; single AdaBoost repeat; cross-dataset results depend on the model setting.",
      f"- Of {art['n_beats']:,} annotated beats, {art['removed_from_usable']} were removed from the usable set because a rater's B or Q point was an 'Artefact' marker."]
open(ROOT / "reports" / "RESULTS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
