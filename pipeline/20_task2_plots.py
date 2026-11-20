"""Step 20 (advisor task 2): plots of the annotated dataset (Python/matplotlib).  Outputs in figures/task2/:
  dataset_overview.png         beats per patient, PEP / RB-vs-RC / C-amplitude / HR / rater agreement
  example_beats.png            ECG + ICG of example beats with Q, R, B (rater 1 and 2), C marks
  features_465_summary.png     the 465 features: mean +- SD of the waveform blocks by dataset, and how often the notch / inflection landmarks exist
  strips/<dataset>_<patient>.png   continuous ECG + ICG strip (10 s) with R, Q, B1, B2, C marks for every patient"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.io import loadmat
from scipy.signal import resample_poly
from joblib import Parallel, delayed
import bioread
from common import *

OUT = FIG / "task2"; (OUT / "strips").mkdir(parents=True, exist_ok=True)
D2 = ROOT / "dataset_task2"
sns.set_theme(style="whitegrid", font_scale=1.0); plt.rcParams.update({"savefig.dpi": 200, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
COL = {"EmpkinS": "#0072B2", "Guardian": "#D55E00"}
tab = pd.read_csv(D2 / "task2_beats.csv"); use = tab[tab.usable == 1]


def overview():
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    s = use.groupby(["dataset", "patient"]).size().reset_index(name="n"); s["lab"] = s.patient.str.replace("GDN", "G").str.replace("VP_", "V")
    ax[0, 0].bar(range(len(s)), s.n, color=[COL[d] for d in s.dataset]); ax[0, 0].set_xticks(range(len(s))); ax[0, 0].set_xticklabels(s.lab, rotation=90, fontsize=7); ax[0, 0].set_ylabel("usable beats"); ax[0, 0].set_title("A  Beats per patient")
    for d, g in use.groupby("dataset"): ax[0, 1].hist(g.PEP_ms_rater1, bins=np.arange(0, 260, 5), alpha=.65, color=COL[d], label=f"{d} (n={len(g)})")
    ax[0, 1].set_xlabel("PEP, rater 1 (ms)"); ax[0, 1].set_ylabel("beats"); ax[0, 1].legend(frameon=False); ax[0, 1].set_title("B  PEP = B - Q")
    for d, g in use.groupby("dataset"): ax[0, 2].scatter(g.C_ms_after_R, g.B_ms_after_R_rater1, s=3, alpha=.25, color=COL[d], label=d)
    rc = np.linspace(60, 330, 50); ax[0, 2].plot(rc, 0.55 * rc + 4.45, "k--", lw=1, label="Lozano 2007 linear (RB = 0.55 RC + 4.45)"); ax[0, 2].legend(frameon=False, fontsize=8, markerscale=4)
    ax[0, 2].set_xlabel("RC: C-point time after R (ms)"); ax[0, 2].set_ylabel("RB: B-point time after R, rater 1 (ms)"); ax[0, 2].set_title("C  B-point vs C-point")
    sns.boxplot(data=use, x="dataset", y="C_amplitude", palette=COL, ax=ax[1, 0], fliersize=1.5); ax[1, 0].set_xlabel(""); ax[1, 0].set_ylabel("C-point amplitude (ICG, dZ/dt)"); ax[1, 0].set_title("D  C-point amplitude")
    for d, g in use.groupby("dataset"): ax[1, 1].hist(g.HR_bpm.dropna(), bins=np.arange(30, 170, 3), alpha=.65, color=COL[d], label=d)
    ax[1, 1].set_xlabel("heart rate (bpm)"); ax[1, 1].set_ylabel("beats"); ax[1, 1].legend(frameon=False); ax[1, 1].set_title("E  Heart rate")
    k = use.PEP_ms_rater2.notna()
    for d, g in use[k].groupby("dataset"): ax[1, 2].scatter(g.PEP_ms_rater1, g.PEP_ms_rater2, s=3, alpha=.25, color=COL[d])
    ax[1, 2].plot([0, 260], [0, 260], "k--", lw=1); mae = (use[k].PEP_ms_rater1 - use[k].PEP_ms_rater2).abs().mean()
    ax[1, 2].set_xlabel("PEP rater 1 (ms)"); ax[1, 2].set_ylabel("PEP rater 2 (ms)"); ax[1, 2].set_title(f"F  Rater agreement (mean |diff| = {mae:.1f} ms)")
    fig.suptitle(f"Annotated ICG/ECG dataset: {use.patient.nunique()} patients, {len(use):,} usable beats", fontweight="bold", y=1.0); fig.tight_layout(); fig.savefig(OUT / "dataset_overview.png"); plt.close(fig)


def examples():
    t = np.arange(-PRE, POST + 1); fig, ax = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    for r, ds in enumerate(["EmpkinS", "Guardian"]):
        g = use[use.dataset == ds]; pats = list(dict.fromkeys(g.patient))[:3] if ds == "EmpkinS" else list(dict.fromkeys(g.patient))[::8][:3]
        for c, p in enumerate(pats):
            z = np.load(D2 / "signals" / f"{ds}_{p}.npz", allow_pickle=True); u = np.where(z["usable"] == 1)[0]; pep = z["PEP_ms_rater1"][u]; i = u[np.argmin(np.abs(pep - np.median(pep)))]
            for k, (key, lab) in enumerate((("ecg", "ECG (cleaned)"), ("icg", "ICG dZ/dt"))):
                a = ax[2 * r + k, c]; y = z[key][i]; a.plot(t, y, color="k", lw=1.4)
                def mark(ms, txt, col, mk, fill=True):
                    if np.isnan(ms): return
                    j = int(round(ms)) + PRE
                    if 0 <= j < len(y): a.plot(ms, y[j], mk, color=col, mfc=col if fill else "none", ms=9, mew=1.8); a.annotate(txt, (ms, y[j]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=col)
                mark(0, "R", "#CC0000", "v"); mark(z["Q_ms_rater1"][i], "Q", "#0072B2", "o")
                if key == "icg":
                    mark(z["B_ms_rater1"][i], "B1", "#D55E00", "s"); mark(z["B_ms_rater2"][i], "B2", "#D55E00", "s", False); mark(z["C_ms"][i], "C", "#009E73", "D")
                a.set_ylabel(lab); a.set_title(f"{ds} {p}, beat {int(z['beat_id'][i])}, {z['phase'][i]}: PEP {z['PEP_ms_rater1'][i]:.0f} ms, RC {z['C_ms'][i]:.0f} ms" if k == 0 else "", fontsize=10)
    for a in ax[-1]: a.set_xlabel("time after R-peak (ms)")
    fig.suptitle("Annotated beats: R-peak (red), Q-onset, B-point (rater 1 filled, rater 2 hollow), C-point", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "example_beats.png"); plt.close(fig)


def features_summary():
    X = np.load(D2 / "features_465.npy"); idx = pd.read_csv(D2 / "features_465_index.csv")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); t = np.arange(151)
    for c, (lo, hi, nm, n) in enumerate([(0, 151, "ICG dZ/dt, R to R+150 ms (features 1-151)", 151), (151, 301, "1st derivative (features 152-301)", 150), (301, 450, "2nd derivative (features 302-450)", 149)]):
        for d in ["EmpkinS", "Guardian"]:
            B = X[(idx.dataset == d).values, lo:hi]; mu, sd = B.mean(0), B.std(0); ax[0, c].plot(np.arange(n), mu, color=COL[d], label=d); ax[0, c].fill_between(np.arange(n), mu - sd, mu + sd, color=COL[d], alpha=.18)
        ax[0, c].set_title(nm, fontsize=10); ax[0, c].set_xlabel("ms after R-peak"); ax[0, c].legend(frameon=False)
    hl = pd.DataFrame({"dataset": idx.dataset, "notch": idx.has_notch, "inflection": idx.has_inflection}); p = hl.groupby("dataset")[["notch", "inflection"]].mean() * 100
    p.plot.bar(ax=ax[1, 0], color=["#56B4E9", "#E69F00"], rot=0); ax[1, 0].set_ylabel("% of beats where the landmark exists"); ax[1, 0].set_title("Last notch / last inflection found (features 451-453, 463-464)", fontsize=10); ax[1, 0].legend(frameon=False)
    names = [l.strip() for l in open(D2 / "features_465_columns.txt")]; HL = names[450:]
    z = pd.DataFrame(X[:, 450:], columns=HL); z["dataset"] = idx.dataset.values
    for d in ["EmpkinS", "Guardian"]: ax[1, 1].hist(z[z.dataset == d].RB_max_2D - 1, bins=np.arange(0, 150, 3), alpha=.65, color=COL[d], label=d)
    ax[1, 1].set_xlabel("time of max 3rd derivative in the search window (ms after R)"); ax[1, 1].set_title("Feature 465: t(max d3Z/dt3)", fontsize=10); ax[1, 1].legend(frameon=False)
    for d in ["EmpkinS", "Guardian"]: ax[1, 2].hist(z[z.dataset == d].RC, bins=np.arange(40, 340, 5), alpha=.65, color=COL[d], label=d)
    ax[1, 2].set_xlabel("RC (ms)"); ax[1, 2].set_title("Feature 462: RC interval", fontsize=10); ax[1, 2].legend(frameon=False)
    fig.suptitle("Sheikh's 465-feature vector (feat_vec_465.m) on the annotated beats", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "features_465_summary.png"); plt.close(fig)


# ---------- continuous strips (need the original continuous signals) ----------
ORIG = {"Pause": "Ruhe", "Valsalva": "Valsalva", "HoldingBreath": "AtmungAnhalten", "TiltUp": "TiltUp", "TiltLevel": "TiltDown"}
def seg_guardian(pid):
    base = GDN / "data_raw" / pid / "tfm_data"; m = loadmat(base / f"{pid.lower()}_no01.mat", struct_as_record=False, squeeze_me=True); names = list(m["IV"].Name); raw = m["RAW_SIGNALS"]
    for ph, orig in ORIG.items():
        if orig not in names: continue
        idx = names.index(orig)
        try: ecg = np.asarray(raw.rawECG2[idx], float); icg = np.asarray(raw.rawICG[idx], float)
        except Exception: continue
        if len(ecg) == 0 or len(icg) == 0: continue
        n = min(len(ecg) // 2, len(icg)); e = ecg[: 2 * n].copy(); ix = np.arange(len(e)); nn = np.isnan(e)
        if nn.any(): e[nn] = np.interp(ix[nn], ix[~nn], e[~nn])
        icg = icg[:n]; nn = np.isnan(icg)
        if nn.any(): icg[nn] = np.interp(np.arange(n)[nn], np.arange(n)[~nn], icg[~nn])
        yield ph, resample_poly(e, 1, 2), icg, 500.0
def seg_empkins(vp):
    for cond in ("tsst", "ftsst"):
        base = EMP / "data_per_subject" / vp / cond / "biopac"; acq = [a for a in sorted((base / "raw").glob("*.acq")) if a.name.lower() == f"biopac_data_{vp.lower()}_{cond}.acq"] or sorted((base / "raw").glob("*.acq"))
        d = bioread.read_file(str(acq[0])); ch = {c.name: c.data for c in d.channels}
        ecg = np.asarray(ch["ECG - ECG100C"], float); icg = np.asarray(ch["ICG - Derivative, NICO100C"], float)
        bad = np.isnan(ecg) | np.isnan(icg)
        if bad.any(): ix = np.arange(len(ecg)); ecg = np.interp(ix, ix[~bad], ecg[~bad]); icg = np.interp(ix, ix[~bad], icg[~bad])
        for ph in ["prep", "pause_1", "talk", "math", "pause_5"]: yield (cond, ph), ecg, icg, 1000.0

def strip(ds, pat):
    g = tab[(tab.dataset == ds) & (tab.patient == pat)]
    try:
        gen = seg_guardian(pat) if ds == "Guardian" else seg_empkins(pat)
        for key, ecg, icg, fs in gen:
            ph = key if ds == "Guardian" else key[1]; cond = "" if ds == "Guardian" else key[0]
            b = g[(g.phase == ph) & ((g.condition.fillna("") == cond))].sort_values("R_sample")
            if len(b) < 10: continue
            ef, i_f = ecg_filter(ecg, fs), icg_filter(icg, fs); r0 = b.R_sample.iloc[0]; a, e = int(r0 - 0.5 * fs), int(r0 + 9.5 * fs); t = (np.arange(a, e) - a) / fs
            fig, ax = plt.subplots(2, 1, figsize=(15, 6.2), sharex=True); ax[0].plot(t, ef[a:e], color="k", lw=.8); ax[1].plot(t, i_f[a:e], color="k", lw=.8)
            bb = b[(b.R_sample >= a) & (b.R_sample < e - 0.6 * fs)]; lab = set()
            def put(a_, x, y, mk, col, name, fill=True):
                a_.plot(x, y, mk, color=col, mfc=col if fill else "none", ms=7, mew=1.5, label=None if name in lab else name); lab.add(name)
            for r in bb.itertuples():
                R = r.R_sample; put(ax[0], (R - a) / fs, ef[R], "v", "#CC0000", "R-peak")
                if not np.isnan(r.Q_ms_after_R_rater1): q = int(R + r.Q_ms_after_R_rater1 * fs / 1000); put(ax[0], (q - a) / fs, ef[q], "o", "#0072B2", "Q onset (rater 1)")
                if not np.isnan(r.B_ms_after_R_rater1): bi = int(R + r.B_ms_after_R_rater1 * fs / 1000); put(ax[1], (bi - a) / fs, i_f[bi], "s", "#D55E00", "B point (rater 1)")
                if not np.isnan(r.B_ms_after_R_rater2): bi = int(R + r.B_ms_after_R_rater2 * fs / 1000); put(ax[1], (bi - a) / fs, i_f[bi], "s", "#D55E00", "B point (rater 2)", False)
                if not np.isnan(r.C_ms_after_R): ci = int(R + r.C_ms_after_R * fs / 1000); put(ax[1], (ci - a) / fs, i_f[ci], "D", "#009E73", "C point")
            ax[0].set_ylabel("ECG (cleaned, a.u.)"); ax[1].set_ylabel("ICG dZ/dt (0.5-40 Hz)"); ax[1].set_xlabel("time (s)")
            for x in ax: x.legend(frameon=False, ncol=4, loc="upper right", fontsize=8)
            ax[0].set_title(f"{ds} {pat}, phase {ph}{' ('+cond+')' if cond else ''}: annotated heartbeats (usable and flagged)  |  PEP = B - Q", fontsize=11)
            fig.tight_layout(); fig.savefig(OUT / "strips" / f"{ds}_{pat}.png", dpi=130); plt.close(fig); return pat
    except Exception as ex:
        print("STRIP FAIL", ds, pat, repr(ex)[:150], flush=True)
    return None


if __name__ == "__main__":
    overview(); examples(); features_summary(); print("overview/examples/features figures done", flush=True)
    pats = tab[["dataset", "patient"]].drop_duplicates().values.tolist()
    res = Parallel(n_jobs=16)(delayed(strip)(d, p) for d, p in pats)
    print(sum(r is not None for r in res), "of", len(pats), "strips")
