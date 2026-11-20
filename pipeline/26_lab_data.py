"""Step 26: data for the website's in-browser Lab (all usable beats, same order as dataset_task2/features_465.npy).
Writes deploy_task1/lab/: meta.json, morph15.f32.gz (n x 15: features 451-465), abel13.f32.gz (n x 13, NaN = rule failed; ms after R), icg404.f32.gz (n x 404: ICG at R-2 .. R+401 ms, 1 kHz).
The ICG window is exactly what Sheikh's extractor needs (R..R+150, C search R+50..R+400, derivative neighbours), so the in-browser code reproduces the stored 465 features exactly: features 1-450 are rebuilt from the ICG window (waveform, 1st and 2nd derivative), 451-465 are shipped."""
import gzip, json
import numpy as np, pandas as pd
from common import *
ABEL = ["arbol2017-isoelectric-crossings", "arbol2017-second-derivative", "arbol2017-third-derivative", "debski1993-second-derivative", "drost2022", "forouzanfar2018",
        "lozano2007-linear-regression", "lozano2007-quadratic-regression", "miljkovic2022", "pale2021", "sherwood1990", "stern1985"]
OUT = ROOT / "deploy_task1" / "lab"; OUT.mkdir(parents=True, exist_ok=True)
m, W, E = load_beats(cohort=True); X = np.load(ROOT / "dataset_task2" / "features_465.npy"); idx = pd.read_csv(ROOT / "dataset_task2" / "features_465_index.csv")
assert len(m) == len(X) == len(idx) and (idx.beat_id.values == m.heartbeat_id.values).all() and (idx.patient.values == m.subject.values).all()
meta_all = read_meta_clean(); coh = (meta_all.B1_ms.notna() & (meta_all.B1_ms - meta_all.Q1_ms).between(20, 250)).values
AX = pd.read_csv(PROC / "abel_real_X.csv")[coh].reset_index(drop=True); assert len(AX) == len(m)
A = np.column_stack([AX[ABEL].values - AX.start_to_R_ms.values[:, None], AX.rr_interval_ms.values]).astype(np.float32)        # rule estimates: ms after R
pats = list(dict.fromkeys(m.subject)); phases = sorted(set(m.phase)); sess = sorted(set(m.condition.fillna("")))
pstart = {p: int(np.where(m.subject.values == p)[0][0]) for p in pats}; pend = {p: int(np.where(m.subject.values == p)[0][-1]) + 1 for p in pats}
meta = dict(n=len(m), cols=dict(sheikh=[l.strip() for l in open(ROOT / "dataset_task2" / "features_465_columns.txt")], abel=ABEL + ["rr_interval_ms"]),
            patients=[dict(id=p, dataset=str(m.dataset[m.subject == p].iloc[0]), start=pstart[p], end=pend[p]) for p in pats], phases=phases, sessions=sess,
            ds=(m.dataset.values == "Guardian").astype(int).tolist(), pat=[pats.index(p) for p in m.subject], phase=[phases.index(p) for p in m.phase], beat_id=m.heartbeat_id.astype(int).tolist(),
            B=m.B1_ms.round(1).tolist(), Q=m.Q1_ms.round(1).tolist(), PEP=m.PEP1.round(1).tolist(), C=m.C_ms.astype(int).tolist(), Camp=m.C_amp.round(4).tolist(),
            B2=[None if pd.isna(v) else round(float(v), 1) for v in m.B2_ms], fs=m.fs.astype(int).tolist())
json.dump(meta, open(OUT / "meta.json", "w"), separators=(",", ":"))
win = W[:, PRE - 2: PRE + 402].astype(np.float32); assert win.shape == (len(m), 404)
def rebuild450(w):                                                   # features 1-450 from the window (R is at index 2)
    d1 = np.gradient(w.astype(np.float64), axis=1); d2 = np.gradient(d1, axis=1)
    return np.concatenate([w[:, 2:153].astype(np.float64), d1[:, 2:152], d2[:, 2:151]], axis=1).astype(np.float32)
assert np.array_equal(rebuild450(win), X[:, :450].astype(np.float32)), "waveform block is not reproducible from the shipped window"
print("check: features 1-450 rebuilt from the shipped window are bit-identical to the stored matrix")
for name, arr in (("morph15.f32.gz", X[:, 450:465].astype(np.float32)), ("abel13.f32.gz", A), ("icg404.f32.gz", win)):
    with gzip.GzipFile(OUT / name, "wb", compresslevel=6, mtime=0) as f: f.write(np.ascontiguousarray(arr).tobytes())
    print(name, arr.shape, f"{(OUT / name).stat().st_size / 1e6:.1f} MB gz")
print("meta.json", f"{(OUT / 'meta.json').stat().st_size / 1e6:.2f} MB")
