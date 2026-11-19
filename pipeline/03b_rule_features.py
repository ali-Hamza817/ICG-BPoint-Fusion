"""Step 3b: classical rule-based B-point estimates (own re-implementations, in the spirit of the 12 algorithms Abel et al. fuse).
NOT identical to PEPbench code. Output = B estimate in ms after R (NaN if rule finds nothing).  Conventions marked ASSUMPTION."""
import numpy as np, pandas as pd
from common import *
R0 = PRE
def rules(w):
    s = w.astype(float); d1 = np.gradient(s); d2 = np.gradient(d1)
    c = R0 + 35 + int(np.argmax(s[R0 + 35: R0 + 351])); out = {}
    ms = lambda i: np.nan if i is None else float(i - R0)
    RC = c - R0
    out["loz_lr"] = 0.55 * RC + 4.45                                     # Lozano 2007 linear (RB from RC)
    out["loz_qr"] = 1.233 * RC - 0.0032 * RC ** 2 - 31.59                # Lozano 2007 quadratic
    a = max(c - 150, R0)
    xs = np.arange(a, c + 1); line = s[a] + (s[c] - s[a]) * (xs - a) / max(c - a, 1)
    out["dro22"] = ms(xs[int(np.argmax(np.abs(line - s[xs])))])           # Drost 2022: max distance to chord C-150ms..C  (ASSUMPTION: absolute distance)
    seg = np.arange(R0 + 10, c)
    mins = [i for i in seg[1:-1] if s[i] < s[i - 1] and s[i] <= s[i + 1]]
    out["ste85"] = ms(mins[-1] if mins else None)                         # Stern 1985: nearest local minimum before C
    zc = [i for i in seg[1:] if s[i - 1] <= 0 < s[i]]
    out["she90"] = ms(zc[-1] if zc else None)                             # Sherwood 1990: nearest zero crossing before C
    m2 = [i for i in seg[1:-1] if d2[i] < d2[i - 1] and d2[i] <= d2[i + 1]]
    out["deb93"] = ms(m2[-1] if m2 else None)                             # Debski 1993: nearest local min of 2nd derivative before C
    mu = s[R0: c + 1].mean(); ic = [i for i in range(R0 + 1, c + 1) if s[i - 1] < mu <= s[i]]
    out["arb17_ic"] = ms(ic[-1] if ic else None)                          # Arbol 2017: last isoelectric (cycle-mean) crossing before C
    w150 = np.arange(max(c - 150, R0), max(c - 100, R0 + 1)); out["arb17_sd"] = ms(w150[int(np.argmax(d1[w150]))]) if len(w150) else np.nan  # peak of d1 in C-150..C-100 (ASSUMPTION)
    return out
if __name__ == "__main__":
    m, W, E = load_beats(cohort=True)
    df = pd.DataFrame([rules(w) for w in W]); df["RC"] = m.C_ms.values
    df.to_csv(PROC / "rule_X.csv", index=False)
    err = df.drop(columns="RC").sub(m.B1_ms.values, axis=0).abs()
    print("rule-based MAE vs rater1 (ms), coverage:"); print(pd.DataFrame({"MAE": err.mean().round(2), "coverage": (1 - df.drop(columns="RC").isna().mean()).round(3)}))
