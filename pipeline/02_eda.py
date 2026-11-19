"""Step 2: classic EDA figures -> figures/eda/*.png and results/eda_summary.json (all numbers computed, none typed)."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from common import *

OUT = FIG / "eda"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
PAL = {"EmpkinS": "#1f77b4", "Guardian": "#d95f02"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 220, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); plt.close(fig)

raw, Wraw, _ = load_beats(cohort=False)
m, W, E = load_beats(cohort=True)
t = np.arange(-PRE, POST + 1)
S = {}
S["raw_beats"] = int(len(raw)); S["cohort_beats"] = int(len(m))
S["raw_by_dataset"] = raw.groupby("dataset").size().to_dict(); S["cohort_by_dataset"] = m.groupby("dataset").size().to_dict()
S["excluded_outlier_beats"] = int((raw.B1_ms.notna() & ~raw.in_cohort).sum()); S["missing_B1"] = int(raw.B1_ms.isna().sum())

# 1 beats per subject / phase ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1.6]})
c = m.groupby(["dataset", "subject"]).size().reset_index(name="n")
sns.barplot(data=c, x="subject", y="n", hue="dataset", palette=PAL, dodge=False, ax=ax[0])
ax[0].set_title("Annotated cardiac cycles per participant (analysis cohort)"); ax[0].set_xlabel(""); ax[0].set_ylabel("cycles")
ax[0].tick_params(axis="x", rotation=90, labelsize=8)
p = m.groupby(["dataset", "phase"]).size().reset_index(name="n")
sns.barplot(data=p, y="phase", x="n", hue="dataset", palette=PAL, ax=ax[1]); ax[1].set_title("Cycles per phase"); ax[1].set_ylabel("")
save(fig, "01_cycles_per_subject_phase")

# 2 demographics ----------------------------------------------------------------------------
try:
    de = pd.read_csv(EMP / "metadata/demographics.csv"); dg = pd.read_csv(GDN / "metadata/demographics.csv")
    S["demo_cols"] = {"emp": list(de.columns), "gdn": list(dg.columns)}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    age_col = lambda d: [c for c in d.columns if c.lower() == "age"][0]
    dg = dg[dg.participant.isin(m[m.dataset == "Guardian"].subject.unique())]; de = de[de.participant.isin(m[m.dataset == "EmpkinS"].subject.unique())]
    for name, d in [("EmpkinS", de), ("Guardian", dg)]:
        sns.histplot(d[age_col(d)], bins=10, color=PAL[name], alpha=.6, label=name, ax=ax[0])
        h = [c for c in d.columns if c.lower().startswith("height")][0]; w = [c for c in d.columns if c.lower().startswith("weight")][0]
        bmi = d[w] / (d[h] / 100) ** 2
        sns.histplot(bmi, bins=10, color=PAL[name], alpha=.6, label=name, ax=ax[1])
        S.setdefault("demographics", {})[name] = dict(n=len(d), age_mean=float(d[age_col(d)].mean()), age_sd=float(d[age_col(d)].std()), bmi_mean=float(bmi.mean()), bmi_sd=float(bmi.std()))
        g = [c for c in d.columns if c.lower() in ("gender", "sex")][0]
        vc = d[g].astype(str).value_counts(); ax[2].bar([f"{name}\n{k}" for k in vc.index], vc.values, color=PAL[name])
    ax[0].set_title("Age (years)"); ax[1].set_title("BMI (kg/m²)"); ax[2].set_title("Sex coding counts (as in file)"); ax[0].legend(); save(fig, "02_demographics")
except Exception as e:
    S["demographics_error"] = repr(e)

# 3 target distribution ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for ds in PAL:
    g = m[m.dataset == ds]
    sns.histplot(g.B1_ms, bins=60, stat="density", color=PAL[ds], alpha=.45, label=ds, ax=ax[0]); sns.kdeplot(g.B1_ms, color=PAL[ds], ax=ax[0])
    sns.histplot(g.PEP1, bins=60, stat="density", color=PAL[ds], alpha=.45, label=ds, ax=ax[1]); sns.kdeplot(g.PEP1, color=PAL[ds], ax=ax[1])
ax[0].set_xlabel("RB interval: B-point after R-peak (ms)"); ax[0].set_title("Target distribution (RB)")
ax[1].set_xlabel("PEP = B − Q-onset (ms)"); ax[1].set_title("Pre-ejection period"); ax[0].legend()
sns.boxplot(data=m, x="dataset", y="B1_ms", palette=PAL, ax=ax[2], width=.5, fliersize=2); ax[2].set_title("RB by dataset"); ax[2].set_ylabel("ms")
save(fig, "03_target_distribution")
for ds, g in m.groupby("dataset"):
    S.setdefault("target_stats", {})[ds] = dict(RB_mean=float(g.B1_ms.mean()), RB_sd=float(g.B1_ms.std()), RB_min=float(g.B1_ms.min()), RB_max=float(g.B1_ms.max()), PEP_mean=float(g.PEP1.mean()), PEP_sd=float(g.PEP1.std()),
        skew=float(stats.skew(g.B1_ms)), kurtosis=float(stats.kurtosis(g.B1_ms)))
ks = stats.ks_2samp(m[m.dataset == "EmpkinS"].B1_ms, m[m.dataset == "Guardian"].B1_ms); S["ks_RB_between_datasets"] = dict(stat=float(ks.statistic), p=float(ks.pvalue))

# 4 per-phase and per-subject --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 2.4]})
sns.violinplot(data=m, y="phase", x="PEP1", hue="dataset", palette=PAL, split=False, cut=0, inner="quartile", density_norm="width", ax=ax[0]); ax[0].set_title("PEP per phase"); ax[0].set_xlabel("PEP (ms)")
order = m.groupby(["dataset", "subject"]).PEP1.median().sort_values().index.get_level_values(1)
sns.boxplot(data=m, x="subject", y="B1_ms", hue="dataset", palette=PAL, order=order, dodge=False, fliersize=1.5, linewidth=.8, ax=ax[1])
ax[1].tick_params(axis="x", rotation=90, labelsize=8); ax[1].set_title("RB interval per participant (sorted by median PEP)"); ax[1].set_ylabel("RB (ms)"); ax[1].set_xlabel("")
save(fig, "04_target_by_phase_subject")
S["between_subject_var_share_RB"] = {ds: float(g.groupby("subject").B1_ms.mean().var() / g.B1_ms.var()) for ds, g in m.groupby("dataset")}

# 5 heart rate -------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for ds in PAL:
    g = m[m.dataset == ds]; sns.histplot(g.HR.dropna(), bins=50, stat="density", color=PAL[ds], alpha=.5, label=ds, ax=ax[0])
    gg = g.dropna(subset=["HR"]); r = stats.spearmanr(gg.HR, gg.B1_ms)
    ax[1].scatter(gg.HR, gg.B1_ms, s=4, alpha=.25, color=PAL[ds], label=f"{ds} (ρ={r.statistic:.2f})")
    S.setdefault("spearman_HR_RB", {})[ds] = float(r.statistic)
    bins = pd.qcut(gg.HR, 15, duplicates="drop"); tr = gg.groupby(bins, observed=True).agg(x=("HR", "median"), y=("B1_ms", "median"))
    ax[1].plot(tr.x, tr.y, color=PAL[ds], lw=2.5, marker="o", ms=4, mec="k")
ax[0].set_title("Heart rate (bpm, from RR)"); ax[0].legend(); ax[1].set_title("RB vs heart rate (lines: binned medians)"); ax[1].set_xlabel("HR (bpm)"); ax[1].set_ylabel("RB (ms)"); ax[1].legend(markerscale=3)
m["HR_group"] = pd.cut(m.HR, [0, 60, 100, 300], labels=["<60", "60–100", ">100"])
sns.boxplot(data=m.dropna(subset=["HR_group"]), x="HR_group", y="B1_ms", hue="dataset", palette=PAL, ax=ax[2], fliersize=1.5); ax[2].set_title("RB by HR group")
save(fig, "05_heart_rate")
S["hr_group_counts"] = m.HR_group.value_counts().to_dict() if False else {str(k): int(v) for k, v in m.HR_group.value_counts().items()}

# 6 inter-rater ------------------------------------------------------------------------------
both = m.dropna(subset=["B2_ms"]).copy(); both["d"] = both.B1_ms - both.B2_ms; both["ad"] = both.d.abs()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for ds in PAL:
    g = both[both.dataset == ds]; ax[0].scatter((g.B1_ms + g.B2_ms) / 2, g.d, s=4, alpha=.3, color=PAL[ds], label=ds)
bias, sd = both.d.mean(), both.d.std()
for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]: ax[0].axhline(y, color="k", ls=ls, lw=1)
ax[0].set_ylim(-120, 120); ax[0].set_title("Bland–Altman, rater 1 vs rater 2 (B-point)"); ax[0].set_xlabel("mean of raters (ms)"); ax[0].set_ylabel("rater1 − rater2 (ms)"); ax[0].legend(markerscale=3)
sns.histplot(both.ad.clip(upper=60), bins=60, color="#555", ax=ax[1]); ax[1].set_title("|rater1 − rater2| (clipped at 60 ms)"); ax[1].set_xlabel("ms")
both["agree"] = pd.cut(both.ad, [-1, 4, 10, 1e9], labels=["high ≤4 ms", "medium 4–10 ms", "low >10 ms"])
vc = both.groupby(["dataset", "agree"], observed=True).size().reset_index(name="n"); vc["pct"] = 100 * vc.n / vc.groupby("dataset").n.transform("sum")
sns.barplot(data=vc, x="agree", y="pct", hue="dataset", palette=PAL, ax=ax[2]); ax[2].set_title("Inter-rater agreement levels"); ax[2].set_ylabel("% of cycles"); ax[2].set_xlabel("")
save(fig, "06_inter_rater_agreement")
S["inter_rater"] = dict(n=int(len(both)), mae=float(both.ad.mean()), sd_abs=float(both.ad.std()), median_abs=float(both.ad.median()), bias=float(bias), loa_low=float(bias - 1.96 * sd), loa_high=float(bias + 1.96 * sd),
    agree_pct={f"{k[0]}|{k[1]}": float(v) for k, v in vc.set_index(["dataset", "agree"]).pct.items()},
    pearson_r=float(stats.pearsonr(both.B1_ms, both.B2_ms)[0]))

# 7 example beats ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
for r_, ds in enumerate(PAL):
    idx = rng.choice(np.where(m.dataset == ds)[0], 4, replace=False)
    for c_, i in enumerate(idx):
        a = axs[r_, c_]; a.plot(t, W[i], color=PAL[ds], lw=1.4)
        a.axvline(0, color="k", ls=":", lw=1); a.axvline(m.B1_ms[i], color="crimson", lw=1.5, label="B (rater 1)")
        if not np.isnan(m.B2_ms[i]): a.axvline(m.B2_ms[i], color="gold", lw=1.5, ls="--", label="B (rater 2)")
        a.axvline(m.C_ms[i], color="green", lw=1, ls="-.", label="C (argmax)"); a.set_title(f"{m.subject[i]} · {m.phase[i]}", fontsize=9)
        a.set_xlim(-100, 450)
axs[0, 0].legend(fontsize=7); axs[1, 0].set_xlabel("ms from R-peak"); fig.suptitle("Example ICG (dZ/dt) cycles with manual B-point labels", y=1.01, fontweight="bold"); save(fig, "07_example_beats")

# 8 average waveforms -----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for ds in PAL:
    k = (m.dataset == ds).values; w = W[k]
    mu, sdv = w.mean(0), w.std(0)
    ax[0].plot(t, mu, color=PAL[ds], label=f"{ds} (n={k.sum()})"); ax[0].fill_between(t, mu - sdv, mu + sdv, color=PAL[ds], alpha=.2)
    # aligned on B
    L = 200; al = np.array([W[j, PRE + int(round(b)) - 100: PRE + int(round(b)) + 200] for j, b in zip(np.where(k)[0], m.B1_ms[k]) if PRE + round(b) - 100 >= 0 and PRE + round(b) + 200 <= W.shape[1]])
    tb = np.arange(-100, 200); ax[1].plot(tb, al.mean(0), color=PAL[ds], label=ds); ax[1].fill_between(tb, al.mean(0) - al.std(0), al.mean(0) + al.std(0), color=PAL[ds], alpha=.2)
    z = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True); ax[2].plot(t, z.mean(0), color=PAL[ds], label=ds)
ax[1].axvline(0, color="crimson"); ax[0].axvline(0, color="k", ls=":")
ax[0].set_title("Mean ± SD ICG aligned on R-peak (raw amplitude)"); ax[1].set_title("Mean ± SD ICG aligned on annotated B-point"); ax[2].set_title("Per-beat z-scored mean, R-aligned")
for a in ax: a.set_xlabel("ms"); a.legend()
ax[0].set_ylabel("dZ/dt (device units)"); save(fig, "08_average_waveforms")
S["amplitude"] = {ds: dict(C_amp_mean=float(g.C_amp.mean()), C_amp_sd=float(g.C_amp.std()), C_amp_median=float(g.C_amp.median())) for ds, g in m.groupby("dataset")}

# 9 waterfall ---------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for a, ds in zip(ax, PAL):
    k = np.where(m.dataset == ds)[0]; o = k[np.argsort(m.B1_ms.values[k])]
    z = W[o][:, PRE - 50: PRE + 351]; vmax = np.percentile(np.abs(z), 99)
    im = a.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-50, 350, len(o), 0])
    a.plot(m.B1_ms.values[o], np.arange(len(o)) + .5, "k.", ms=1.2); a.set_title(f"{ds}: ICG cycles sorted by B-point (black = B)"); a.set_xlabel("ms from R-peak"); a.set_ylabel("cycle rank")
    plt.colorbar(im, ax=a, fraction=.04)
save(fig, "09_waterfall_sorted_by_B")

# 10 Lozano relationship RB vs RC and morphology ---------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for ds in PAL:
    g = m[m.dataset == ds]; ax[0].scatter(g.C_ms, g.B1_ms, s=4, alpha=.25, color=PAL[ds], label=ds)
    lr = stats.linregress(g.C_ms, g.B1_ms); S.setdefault("RB_vs_RC_linreg", {})[ds] = dict(slope=float(lr.slope), intercept=float(lr.intercept), r2=float(lr.rvalue ** 2))
    xs = np.linspace(g.C_ms.min(), g.C_ms.max(), 50); ax[0].plot(xs, lr.intercept + lr.slope * xs, color=PAL[ds], lw=2)
    sns.histplot(g.C_ms, bins=50, color=PAL[ds], alpha=.5, ax=ax[1], stat="density"); sns.histplot(g.C_amp, bins=50, color=PAL[ds], alpha=.5, ax=ax[2], stat="density")
ax[0].set_xlabel("RC interval (ms)"); ax[0].set_ylabel("RB interval (ms)"); ax[0].set_title("RB vs RC (Lozano-type relation)"); ax[0].legend(markerscale=3)
ax[1].set_title("RC interval (C-point latency)"); ax[1].set_xlabel("ms"); ax[2].set_title("C-point amplitude"); ax[2].set_xlabel("dZ/dt (device units, NOT comparable across datasets)")
save(fig, "10_RB_vs_RC_and_C_point")

# 11 correlation heatmap -----------------------------------------------------------------------
cols = ["B1_ms", "Q1_ms", "PEP1", "C_ms", "C_amp", "RR_prev_ms", "HR"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a, ds in zip(ax, PAL):
    cm = m[m.dataset == ds][cols].corr(method="spearman"); sns.heatmap(cm, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, cbar=False, ax=a); a.set_title(f"{ds}: Spearman correlation")
save(fig, "11_correlation_heatmaps")

# 12 signal quality: baseline noise, per-subject amplitude ---------------------------------------
m["pre_noise"] = W[:, :PRE - 20].std(1)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
sns.boxplot(data=m, x="subject", y="C_amp", hue="dataset", palette=PAL, dodge=False, ax=ax[0], fliersize=1); ax[0].tick_params(axis="x", rotation=90, labelsize=7); ax[0].set_title("C-point amplitude per participant"); ax[0].set_xlabel("")
sns.violinplot(data=m, x="dataset", y="pre_noise", palette=PAL, cut=0, ax=ax[1]); ax[1].set_title("Pre-R segment SD (noise proxy)"); ax[1].set_ylabel("SD (device units)")
save(fig, "12_amplitude_and_noise")

# 13 outliers overview + raw vs cohort ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for ds in PAL:
    g = raw[raw.dataset == ds]; sns.histplot(g.PEP1.dropna().clip(-450, 450), bins=120, color=PAL[ds], alpha=.5, label=ds, ax=ax[0])
ax[0].axvspan(20, 250, color="green", alpha=.08, label="cohort window"); ax[0].set_title("PEP (B−Q) before cohort filter: annotation outliers"); ax[0].set_xlabel("ms (clipped ±450)"); ax[0].legend()
o = raw[raw.B1_ms.notna() & ~raw.in_cohort].groupby(["subject", "condition", "phase"]).size().sort_values(ascending=False).head(8)
ax[1].barh([" ".join(map(str, k)) for k in o.index][::-1], o.values[::-1], color="#a33"); ax[1].set_title("Where the excluded beats come from"); ax[1].set_xlabel("beats")
save(fig, "13_cohort_filter_outliers")
S["excluded_top_segments"] = {" ".join(map(str, k)): int(v) for k, v in o.items()}

# 14 QQ + normality of target ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for a, ds in zip(ax, PAL):
    stats.probplot(m[m.dataset == ds].B1_ms, dist="norm", plot=a); a.get_lines()[0].set_color(PAL[ds]); a.get_lines()[0].set_markersize(2); a.set_title(f"{ds}: QQ-plot of RB")
save(fig, "14_qq_plots")

json.dump(S, open(RES / "eda_summary.json", "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float)[:6000])
