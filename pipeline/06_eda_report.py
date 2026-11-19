"""Step 6: write reports/EDA_REPORT.md from results/eda_summary.json (numbers are copied programmatically, never typed)."""
import json
from common import *
S = json.load(open(RES / "eda_summary.json")); (ROOT / "reports").mkdir(exist_ok=True)
F = lambda n, cap: f"![{cap}](../figures/eda/{n}.png)\n\n*{cap}*\n"
td, dm, ir = S["target_stats"], S["demographics"], S["inter_rater"]
L = ["# Exploratory data analysis: EmpkinS + Guardian (beat-to-beat ICG B-point)\n",
     "All numbers are computed by `pipeline/02_eda.py` and stored in `results/eda_summary.json`. Labels = rater 1; rater 2 is used only for agreement.\n",
     "## 1. Cohort\n",
     f"- Annotated cycles built from the raw files: **{S['raw_beats']:,}**; analysis cohort: **{S['cohort_beats']:,}** "
     f"(EmpkinS {S['cohort_by_dataset']['EmpkinS']:,}, Guardian {S['cohort_by_dataset']['Guardian']:,}).",
     f"- Excluded: {S['excluded_outlier_beats']} cycles with physiologically impossible labels (PEP = B − Q outside 20–250 ms) and {S['missing_B1']} cycle without a B-point label. "
     "Largest sources: " + ", ".join(f"{k} ({v})" for k, v in list(S["excluded_top_segments"].items())[:3]) + ".",
     "- Abel et al. report 11,239 cycles after their own exclusions; ours differ because their exclusion rules were not replicated.\n",
     "| Dataset | Participants | Age (y) | BMI (kg/m²) |", "|---|---|---|---|"]
for k, v in dm.items(): L.append(f"| {k} | {v['n']} | {v['age_mean']:.2f} ± {v['age_sd']:.2f} | {v['bmi_mean']:.2f} ± {v['bmi_sd']:.2f} |")
L += ["", F("01_cycles_per_subject_phase", "Cycles per participant and per phase"), F("02_demographics", "Demographics"),
      "## 2. Target (B-point) distribution\n", "| Dataset | RB mean ± SD (ms) | min / max | PEP mean ± SD (ms) | skew | excess kurtosis |", "|---|---|---|---|---|---|"]
for k, v in td.items(): L.append(f"| {k} | {v['RB_mean']:.1f} ± {v['RB_sd']:.1f} | {v['RB_min']:.0f} / {v['RB_max']:.0f} | {v['PEP_mean']:.1f} ± {v['PEP_sd']:.1f} | {v['skew']:.2f} | {v['kurtosis']:.2f} |")
L += [f"\nKolmogorov–Smirnov test of RB between datasets: D = {S['ks_RB_between_datasets']['stat']:.3f}, p = {S['ks_RB_between_datasets']['p']:.2g} "
      "(the two datasets have clearly different target ranges, the cause of cross-dataset error).\n",
      f"Share of RB variance explained by between-participant means: " + ", ".join(f"{k} {v:.2f}" for k, v in S["between_subject_var_share_RB"].items()) + ".\n",
      F("03_target_distribution", "Distribution of the RB interval and PEP"), F("04_target_by_phase_subject", "B-point by phase and by participant"), F("14_qq_plots", "QQ plots of RB"),
      "## 3. Heart rate\n", "Spearman ρ between HR and RB: " + ", ".join(f"{k} {v:.2f}" for k, v in S["spearman_HR_RB"].items()) + ". "
      "HR-group counts: " + ", ".join(f"{k}: {v}" for k, v in S["hr_group_counts"].items()) + ".\n", F("05_heart_rate", "Heart rate and its relation to RB"),
      "## 4. Inter-rater agreement\n",
      f"n = {ir['n']:,} cycles with both raters. MAE {ir['mae']:.2f} ms (SD of |diff| {ir['sd_abs']:.2f}), median |diff| {ir['median_abs']:.1f} ms, bias {ir['bias']:.2f} ms, "
      f"limits of agreement {ir['loa_low']:.1f} to {ir['loa_high']:.1f} ms, Pearson r = {ir['pearson_r']:.3f}. (Abel et al. report 8.35 ± 14.78 ms.)\n",
      "| Dataset | agreement level | % of cycles |", "|---|---|---|"] + [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v:.1f} |" for k, v in ir["agree_pct"].items()] + [
      "", F("06_inter_rater_agreement", "Bland–Altman, histogram and agreement levels"),
      "## 5. Waveform morphology\n", F("07_example_beats", "Example cycles with both raters' B-points"), F("08_average_waveforms", "Average waveforms aligned on R and on B"), F("09_waterfall_sorted_by_B", "All cycles sorted by B-point"),
      "## 6. Relations between variables\n", "Linear fit RB ~ RC: " + "; ".join(f"{k}: slope {v['slope']:.2f}, intercept {v['intercept']:.1f}, R² {v['r2']:.2f}" for k, v in S["RB_vs_RC_linreg"].items()) + ".\n",
      F("10_RB_vs_RC_and_C_point", "RB vs RC and C-point statistics"), F("11_correlation_heatmaps", "Spearman correlations"),
      "## 7. Signal quality and outliers\n", "C-point amplitude (device units, not comparable between datasets): " + "; ".join(f"{k} median {v['C_amp_median']:.2f}" for k, v in S["amplitude"].items()) + ".\n",
      F("12_amplitude_and_noise", "Amplitude and noise proxy"), F("13_cohort_filter_outliers", "Excluded annotation outliers")]
(ROOT / "reports" / "EDA_REPORT.md").write_text("\n".join(L))
print("written", len(L), "blocks")
