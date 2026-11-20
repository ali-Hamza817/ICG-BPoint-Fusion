"""Step 22: flag label points that are 'Artefact' markers, not real annotations.
Both raters can mark a beat as 'Artefact' (channel 'Artefact', label 'Artefact', same sample in the ICG and ECG label file). step 1 (common.read_labels) read every non-heartbeat
row as a B or Q point, so for these beats B/Q silently became the artefact sample.  This script compares each beat's B1/Q1/B2/Q2 sample with the artefact samples of the same
file (dataset, subject, condition, phase, rater) and writes data/processed/artefact_flags.csv (row-aligned with beats_meta.csv).  common.load_beats() then sets the affected
B/Q values to NaN (so PEP and in_cohort drop them).  Also writes results/artefact_summary.json."""
import glob, json, re
import numpy as np, pandas as pd
from common import *
m = pd.read_csv(PROC / "beats_meta.csv"); art = {}
PH = {"pause": "Pause", "valsalva": "Valsalva", "holdingbreath": "HoldingBreath", "tiltup": "TiltUp", "tiltlevel": "TiltLevel"}
def key(path, ds):
    rater = "rater_02" if "rater_02" in path else "rater_01"
    if ds == "EmpkinS":
        mm = re.search(r"per_subject/(VP_\d+)/(\w+)/biopac.*reference_labels_vp_\d+_\w+?_(prep|pause_1|talk|math|pause_5)_(icg|ecg)\.csv", path); return ("EmpkinS", mm.group(1), mm.group(2), mm.group(3), rater)
    mm = re.search(r"data_raw/(GDN\d+)/tfm_data.*reference_labels_gdn\d+_(\w+)_(icg|ecg)\.csv", path); return ("Guardian", mm.group(1), "", PH[mm.group(2)], rater)
for ds, pat in (("EmpkinS", str(EMP / "data_per_subject/*/*/biopac/reference_labels/rater_0*/*.csv")), ("Guardian", str(GDN / "data_raw/*/tfm_data/reference_labels/rater_0*/*.csv"))):
    for f in glob.glob(pat):
        try: k = key(f, ds)
        except Exception: continue
        art.setdefault(k, set()).update(pd.read_csv(f).query("channel == 'Artefact'").sample_absolute.astype(int).tolist())
def flag(col, rater):
    out = []
    for r in m.itertuples():
        s = art.get((r.dataset, r.subject, "" if pd.isna(r.condition) else r.condition, r.phase, rater), set()); v = getattr(r, col)
        out.append((not pd.isna(v)) and int(v) in s)
    return np.array(out)
F = pd.DataFrame({"art_B1": flag("B1", "rater_01"), "art_Q1": flag("Q1", "rater_01"), "art_B2": flag("B2", "rater_02"), "art_Q2": flag("Q2", "rater_02")})
F.to_csv(PROC / "artefact_flags.csv", index=False)
old = (m.B1_ms.notna() & (m.B1_ms - m.Q1_ms).between(20, 250)); new = old & ~(F.art_B1 | F.art_Q1)
rep = dict(n_beats=len(m), rater1_beats_with_artefact_B_or_Q=int((F.art_B1 | F.art_Q1).sum()), rater2_beats_with_artefact_B_or_Q=int((F.art_B2 | F.art_Q2).sum()),
           usable_before=int(old.sum()), usable_after=int(new.sum()), removed_from_usable=int((old & ~new).sum()), by_dataset_after=m[new].groupby("dataset").size().to_dict())
json.dump(rep, open(RES / "artefact_summary.json", "w"), indent=1); print(json.dumps(rep, indent=1))
