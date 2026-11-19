"""Step 8: re-implementation of Abel et al. 2026 Table 1 rule features (11 of the 12 PEPbench algorithms) + RR interval.
NOT PEPbench/biopsykit code (could not be installed: PyPI ~2 KB/s on this server). Written from the Table-1 descriptions;
every guess is marked ASSUMPTION. Omitted: Miljkovic2022 (description in Table 1 too vague to implement honestly).
Output: data/processed/abel11_X.csv (11 B estimates in ms after R, NaN = rule found nothing; same row order as cohort_meta.csv)."""
import numpy as np, pandas as pd
from common import *
R0 = PRE


def abel_rules(w):
    s = w.astype(float); d1 = np.gradient(s); d2 = np.gradient(d1); d3 = np.gradient(d2)
    c = R0 + 50 + int(np.argmax(s[R0 + 50: R0 + 401]))                   # C = max of ICG in R+50..R+400 ms
    ms = lambda i: np.nan if i is None else float(i - R0)
    out = {}
    RC = c - R0
    out["loz_lr"] = 0.55 * RC + 4.45                                       # Lozano 2007 linear (Table 1)
    out["loz_qr"] = 1.233 * RC - 0.0032 * RC ** 2 - 31.59                  # Lozano 2007 quadratic (Table 1)
    a = max(c - 150, R0)
    xs = np.arange(a, c + 1); line = s[a] + (s[c] - s[a]) * (xs - a) / max(c - a, 1)
    out["dro22"] = ms(xs[int(np.argmax(np.abs(line - s[xs])))])             # Drost 2022: max distance to chord C-150ms..C (ASSUMPTION: absolute distance)
    seg = np.arange(R0 + 10, c)
    mins = [i for i in seg[1:-1] if s[i] < s[i - 1] and s[i] <= s[i + 1]]
    out["ste85"] = ms(mins[-1] if mins else None)                           # Stern 1985: local minimum nearest C
    m2 = [i for i in seg[1:-1] if d2[i] < d2[i - 1] and d2[i] <= d2[i + 1]]
    out["deb93"] = ms(m2[-1] if m2 else None)                               # Debski 1993: local min of 2nd derivative nearest C (ASSUMPTION: 2nd-derivative variant)
    zc = [i for i in seg[1:] if s[i - 1] <= 0 < s[i]]
    out["she90"] = ms(zc[-1] if zc else None)                               # Sherwood 1990: zero crossing nearest C
    mu = s[R0: c + 1].mean(); ic = [i for i in range(R0 + 1, c + 1) if s[i - 1] < mu <= s[i]]
    out["arb17_ic"] = ms(ic[-1] if ic else None)                            # Arbol 2017 IC: last crossing of cycle mean before C
    w1 = np.arange(max(c - 150, R0), max(c - 100, R0 + 1))
    out["arb17_sd"] = ms(w1[int(np.argmax(d2[w1]))]) if len(w1) else np.nan  # Arbol 2017 SD: peak of 2nd derivative in C-150..C-100 (ASSUMPTION: derivative order from name)
    w2 = np.arange(max(c - 300, R0), c)
    out["arb17_td"] = ms(w2[int(np.argmax(d3[w2]))]) if len(w2) > 2 else np.nan  # Arbol 2017 TD: peak of 3rd derivative in C-300..C (ASSUMPTION)
    # Forouzanfar 2018: A point = max in R-50..R+60 (ASSUMPTION), most prominent rising segment between A and C,
    # B = local max / zero crossing nearest C inside it, else start of segment
    A = R0 - 50 + int(np.argmax(s[R0 - 50: R0 + 61]))
    if c - A > 10:
        up = d1[A:c] > 0; best, cur, bs, be = 0.0, None, None, None
        for k, flag in enumerate(up):
            if flag and cur is None: cur = k
            if (not flag or k == len(up) - 1) and cur is not None:
                e = k if not flag else k + 1; gain = s[A + e - 1] - s[A + cur]
                if gain > best: best, bs, be = gain, A + cur, A + e - 1
                cur = None
        if bs is not None:
            cand = [i for i in range(bs + 1, be) if (s[i] >= s[i - 1] and s[i] > s[i + 1]) or (s[i - 1] <= 0 < s[i])]
            out["for18"] = ms(cand[-1] if cand else bs)
        else: out["for18"] = np.nan
    else: out["for18"] = np.nan
    # Pale 2021: search window C-80ms .. last time the signal exceeds C/2 before C; closest local min before C;
    # fallback: window minimum (ASSUMPTION: the slope-threshold branch (0.11) is omitted -> units unspecified)
    half = 0.5 * s[c]; above = np.where(s[R0: c] > half)[0]
    lo = max(c - 80, R0); hi = c if len(above) == 0 else R0 + int(above[0])  # first crossing = start of the main upstroke
    hi = max(hi, lo + 3)
    wi = np.arange(lo, min(hi, c))
    lm = [i for i in wi[1:-1] if s[i] < s[i - 1] and s[i] <= s[i + 1]]
    out["pal21"] = ms(lm[-1] if lm else (int(wi[np.argmin(s[wi])]) if len(wi) else None))
    return out


if __name__ == "__main__":
    m, W, E = load_beats(cohort=True)
    df = pd.DataFrame([abel_rules(w) for w in W])
    df.to_csv(PROC / "abel11_X.csv", index=False)
    err = df.sub(m.B1_ms.values, axis=0).abs()
    t = pd.DataFrame({"MAE_vs_rater1": err.mean().round(2), "coverage": (1 - df.isna().mean()).round(3)})
    print(t); print("beats with >=1 NaN:", int(df.isna().any(axis=1).sum()), "of", len(df))
    t.to_csv(RES / "table_abel11_rule_mae.csv")
