"""Lab API for the in-browser experiments (runs inside Pyodide, in your browser; nothing is sent to a server).
Everything here is plain Python/numpy and is shown in the editor, so each step can be read and changed.

    lab.features("sheikh" | "waveform" | "morphology" | "abel" | "fusion")  -> feature matrix, one row per usable beat
    lab.y                         B-point time after the R-peak (ms), rater 1  (the target)
    lab.subject / lab.dataset     patient index / 0 = EmpkinS, 1 = Guardian for every beat
    lab.subject_folds(k, seed)    subject-wise folds: a patient is never in training and test at once
    lab.train_test("EmpkinS", "Guardian") / lab.holdout("EmpkinS_VP_003")
    lab.progress(done, total, text) and lab.report(name, y_true, y_pred, **info)  -> shows the result in the page
"""
import json, time
import numpy as np
import js
from pyodide.ffi import to_js

def _post(kind, **kw):
    js.postMessage(to_js(dict(type=kind, **kw), dict_converter=js.Object.fromEntries, create_pyproxies=False))

META = json.loads(_meta_json)                                               # noqa: F821  (set by the page)
ICG = np.frombuffer(_icg.to_bytes(), dtype=np.float32).reshape(META["n"], 404).copy()      # noqa: F821  R-2 .. R+401 ms
MORPH = np.frombuffer(_morph.to_bytes(), dtype=np.float32).reshape(META["n"], 15).copy()   # noqa: F821  features 451-465
ABEL = np.frombuffer(_abel.to_bytes(), dtype=np.float32).reshape(META["n"], 13).copy()     # noqa: F821  12 rules (ms after R) + RR


def _build_sheikh():
    w = ICG.astype(np.float64); d1 = np.gradient(w, axis=1); d2 = np.gradient(d1, axis=1)
    return np.concatenate([ICG[:, 2:153].astype(np.float64), d1[:, 2:152], d2[:, 2:151], MORPH.astype(np.float64)], axis=1).astype(np.float32)


class Lab:
    def __init__(self):
        self.X465 = _build_sheikh(); self.n = META["n"]
        self.names = dict(sheikh=META["cols"]["sheikh"], abel=META["cols"]["abel"])
        self.y = np.array(META["B"], dtype=np.float64); self.subject = np.array(META["pat"]); self.dataset = np.array(META["ds"])
        self.patients = [p["id"] for p in META["patients"]]; self._abel = None

    def _abel_imputed(self):                                               # Abel's exp.-2 style: missing rule estimate = median of the other rules in the same beat
        if self._abel is None:
            A = ABEL.astype(np.float64).copy(); med = np.nanmedian(A[:, :12], axis=1)
            for j in range(12): A[:, j] = np.where(np.isnan(A[:, j]), med, A[:, j])
            rr = A[:, 12]; pm = np.array([np.nanmedian(rr[self.subject == s]) for s in range(len(self.patients))])
            A[:, 12] = np.where(np.isnan(rr), pm[self.subject], rr); A[np.isnan(A)] = 0.0
            self._abel = A.astype(np.float32)
        return self._abel

    def features(self, which="sheikh"):
        X = self.X465
        return {"sheikh": X, "waveform": X[:, :450], "morphology": X[:, 450:], "abel": self._abel_imputed(), "fusion": np.hstack([X, self._abel_imputed()])}[which]

    def subject_folds(self, k=5, seed=0):
        rng = np.random.default_rng(seed); sizes = np.bincount(self.subject, minlength=len(self.patients)); load = np.zeros(k); fold = {}
        for s in sorted(rng.permutation(len(self.patients)), key=lambda s: -sizes[s] + rng.normal(0, 30)):
            f = int(np.argmin(load)); fold[int(s)] = f; load[f] += sizes[s]
        fid = np.array([fold[int(s)] for s in self.subject])
        return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)]

    def train_test(self, train="EmpkinS", test="Guardian"):
        d = {"EmpkinS": 0, "Guardian": 1}; return np.where(self.dataset == d[train])[0], np.where(self.dataset == d[test])[0]

    def holdout(self, patient_id):
        i = self.patients.index(patient_id); te = np.where(self.subject == i)[0]; return np.where(self.subject != i)[0], te

    def progress(self, done, total, text=""):
        _post("progress", done=int(done), total=int(total), text=str(text))

    def report(self, name, y_true, y_pred, **info):
        ok = ~np.isnan(y_pred); e = y_pred[ok] - y_true[ok]; ae = np.abs(e); per = []
        for i in range(len(self.patients)):
            k = ok & (self.subject == i)
            if k.any(): per.append([i, float(np.abs(y_pred[k] - y_true[k]).mean()), int(k.sum())])
        idx = np.where(ok)[0]; step = max(1, len(idx) // 1200)
        sc = np.column_stack([y_true[idx[::step]], y_pred[idx[::step]]]).round(1).tolist()
        h, _ = np.histogram(np.clip(e, -60, 60), bins=np.arange(-60, 61, 4))
        _post("result", result=dict(name=str(name), n=int(ok.sum()), MAE=float(ae.mean()), SD=float(ae.std()), MedAE=float(np.median(ae)), ME=float(e.mean()),
                                    subjMAE=float(np.mean([p[1] for p in per])), per=per, scatter=sc, hist=h.tolist(), info={k: str(v) for k, v in info.items()}))


lab = Lab()
_post("data_ready", n=lab.n)
