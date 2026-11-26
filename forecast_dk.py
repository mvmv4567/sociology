"""
Religion in Europe, 2036: steps 2 and 3 (forecast with ranges, back-cast).

Run (same folder as religion_dk.py):
  python forecast_dk.py path/to/ZA7503_v3-0-0.dta path/to/Denmark-2036.xlsx

Inputs
  ZA7503_v3-0-0.dta : EVS Trend File 1981-2017 (GESIS), Denmark only is used
  Denmark-2036.xlsx : UN World Population Prospects 2024 (medium variant),
                      Denmark 2036, population by 5-year age group and sex

Outputs
  forecast_2036_all.csv      every scenario, with bootstrap interval
  forecast_2036_summary.csv  central value and ranges per indicator
  backcast.csv               predictions of 2008 and 2017 from earlier waves
  backcast_summary.csv       errors per scenario and benchmark
  forecast_chart.png         (only if matplotlib is installed)

Method in short
  Birth cohorts (10 years) are the unit. Three scenarios:
   S1 cohort + age     : cohorts keep their level, people age, no period drift
   S2 cohort + period  : cohorts keep their level, plus a common linear
                         period trend continued to the target year
   S3 trend            : overall share by year (logit-linear), no cohorts
  Cohorts that are not observed (or have n < 30) in the fitting data get
  their level from an assumption:
   same_as_last   : equal to the youngest estimated cohort
   trend_continues: the cohort-to-cohort slope of the last estimated
                    cohorts (up to three) is continued
  2017 is run two ways: all respondents, and interviewer-administered
  (CAPI) respondents only, because Denmark used mixed modes in 2017.
  Uncertainty: stratified bootstrap over respondents (B = 200).
"""
import sys
import numpy as np
import pandas as pd
from religion_dk import indicators, COUNTRY

SEED = 2036
B = 200
MIN_N = 30
TARGET_YEAR = 2036
IND_PRIMARY = ["believing_god", "belonging", "practice_monthly"]
IND_ALL = ["believing_god", "believing_importance", "believing_religious_person",
           "belonging", "practice_monthly", "practice_weekly"]

ORDER = ["<=1939", "1940-49", "1950-59", "1960-69", "1970-79",
         "1980-89", "1990-99", "2000+"]
BINS = [-np.inf, 1939, 1949, 1959, 1969, 1979, 1989, 1999, np.inf]
CENTER = {"<=1939": 1930.0, "1940-49": 1944.5, "1950-59": 1954.5,
          "1960-69": 1964.5, "1970-79": 1974.5, "1980-89": 1984.5,
          "1990-99": 1994.5, "2000+": 2009.0}  # 2000+ = born 2000-2018

SCEN = {"age": "S1_cohort_age", "period": "S2_cohort_period", "trend": "S3_trend"}
ASSUMP = ["same_as_last", "trend_continues"]


# ---------------------------------------------------------------- data
def cohort_of(birth):
    return pd.cut(birth, BINS, labels=ORDER).astype(str)


def load(path):
    cols = ["S002EVS", "S003", "S020", "S017", "X002", "mode",
            "F050", "F063", "F034", "F025", "F028"]
    df = pd.read_stata(path, convert_categoricals=False, columns=cols)
    df = df[df["S003"] == COUNTRY].copy()
    df = df.rename(columns={"S002EVS": "wave", "S020": "year", "S017": "w"})
    df.loc[df["X002"] < 1800, "X002"] = np.nan
    df = pd.concat([df, indicators(df)], axis=1)
    return df


def prepare(df):
    d = df.dropna(subset=["X002"]).copy()
    d["cohort"] = cohort_of(d["X002"])
    d["age"] = d["year"] - d["X002"]
    return d


def population_2036(xlsx):
    t = pd.read_excel(xlsx)
    rows = []
    for _, r in t.iterrows():
        lab = str(r["Age"]).strip()
        total = float(r["M"] + r["F"])
        if lab.endswith("+"):
            ages = [int(lab[:-1])]
        else:
            lo, hi = (int(x) for x in lab.split("-"))
            ages = list(range(lo, hi + 1))
        for a in ages:
            rows.append((a, total / len(ages)))
    pop = pd.DataFrame(rows, columns=["age", "pw"])
    pop = pop[pop["age"] >= 18].copy()  # EVS interviews adults only
    pop["year"] = TARGET_YEAR
    pop["cohort"] = cohort_of(TARGET_YEAR - pop["age"])
    return pop


def wshare(d, ind):
    x = d[[ind, "w"]].dropna()
    return float(np.average(x[ind], weights=x["w"])) if len(x) else np.nan


# ---------------------------------------------------------------- model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def irls(X, y, w, ridge=1e-4, iters=60):
    beta = np.zeros(X.shape[1])
    pen = ridge * np.eye(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = sigmoid(eta)
        v = np.maximum(p * (1 - p), 1e-9)
        z = eta + (y - p) / v
        W = w * v
        new = np.linalg.solve(X.T @ (X * W[:, None]) + pen, X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def design(d, est, kind):
    cols = [np.ones(len(d))]
    if kind in ("age", "period"):
        for c in est[1:]:
            cols.append((d["cohort"].to_numpy() == c).astype(float))
    if kind == "age":
        cols.append((d["age"].clip(18, 85).to_numpy() - 45) / 10)
    else:  # period and trend
        cols.append((d["year"].to_numpy() - 2000) / 10)
    return np.column_stack(cols)


def fit(train, ind, kind):
    d = train[[ind, "w", "cohort", "age", "year"]].dropna(subset=[ind])
    counts = d["cohort"].value_counts()
    est = [c for c in ORDER if counts.get(c, 0) >= MIN_N]
    X = design(d, est, kind)
    w = d["w"].to_numpy()
    w = w / w.mean()
    beta = irls(X, d[ind].to_numpy(float), w)
    return {"kind": kind, "est": est, "beta": beta}


def cohort_coef(m, c, assumption):
    est, beta = m["est"], m["beta"]
    coefs = {est[0]: 0.0}
    for i, e in enumerate(est[1:], start=1):
        coefs[e] = beta[i]
    if c in coefs:
        return coefs[c]
    last = est[-1]
    if ORDER.index(c) > ORDER.index(last):
        base = coefs[last]
        if assumption == "same_as_last":
            return base
        use = est[-3:]
        if len(use) < 2:
            return base
        slope = np.polyfit([CENTER[u] for u in use], [coefs[u] for u in use], 1)[0]
        return base + slope * (CENTER[c] - CENTER[last]) / 10
    near = min(est, key=lambda e: abs(CENTER[e] - CENTER[c]))
    return coefs[near]


def predict(m, target, year, assumption):
    b = m["beta"]
    if m["kind"] == "trend":
        return float(sigmoid(b[0] + b[1] * (year - 2000) / 10))
    cc = target["cohort"].map({c: cohort_coef(m, c, assumption)
                               for c in target["cohort"].unique()}).to_numpy()
    if m["kind"] == "age":
        eta = b[0] + cc + b[-1] * (target["age"].clip(18, 85).to_numpy() - 45) / 10
    else:
        eta = b[0] + cc + b[-1] * (year - 2000) / 10
    return float(np.average(sigmoid(eta), weights=target["pw"]))


def all_predictions(train, target, year, ind):
    out = {}
    for kind, name in SCEN.items():
        m = fit(train, ind, kind)
        if kind == "trend":
            out[(name, "-")] = predict(m, target, year, "-")
        else:
            for a in ASSUMP:
                out[(name, a)] = predict(m, target, year, a)
    return out


def resample(train, rng):
    parts = []
    for _, g in train.groupby("wave"):
        parts.append(g.iloc[rng.integers(0, len(g), len(g))])
    return pd.concat(parts)


def scenario_table(train, target, year, ind, rng, label_cols):
    point = all_predictions(train, target, year, ind)
    boots = {k: [] for k in point}
    for _ in range(B):
        res = all_predictions(resample(train, rng), target, year, ind)
        for k, v in res.items():
            boots[k].append(v)
    rows = []
    for (scen, a), p in point.items():
        lo, hi = np.percentile(boots[(scen, a)], [2.5, 97.5])
        rows.append(dict(label_cols, indicator=ind, scenario=scen,
                         post2000=a, point=p, lo=lo, hi=hi))
    return rows


# ---------------------------------------------------------------- runs
def run_forecast(df, pop, rng):
    d = prepare(df)
    variants = {"all2017": d,
                "capi2017": d[~((d["wave"] == 5) & (d["mode"] != 1))]}
    rows = []
    for vname, train in variants.items():
        for ind in IND_ALL:
            rows += scenario_table(train, pop, TARGET_YEAR, ind, rng,
                                   {"variant": vname})
        print("forecast done:", vname)
    return pd.DataFrame(rows)


def run_backcast(df, rng):
    d = prepare(df)
    rows = []
    for train_waves, tw in [([1, 2, 3, 4], 5), ([1, 2, 3], 4)]:
        train = d[d["wave"].isin(train_waves)]
        tgt_all = d[d["wave"] == tw].copy()
        tgt_all["pw"] = tgt_all["w"]
        year = int(tgt_all["year"].iloc[0])
        last_wave = d[d["wave"] == train_waves[-1]]
        for ind in IND_ALL:
            rs = scenario_table(train, tgt_all, year, ind, rng,
                                {"target_year": year})
            act_all = wshare(tgt_all, ind)
            act_capi = wshare(tgt_all[tgt_all["mode"] == 1], ind) if tw == 5 else np.nan
            locf = wshare(last_wave, ind)
            for r in rs:
                r.update(actual_all=act_all, actual_capi=act_capi, locf=locf)
            rows += rs
        print("backcast done: target", year)
    return pd.DataFrame(rows)


def summarize_forecast(fc):
    out = []
    for ind, g in fc.groupby("indicator"):
        out.append(dict(
            indicator=ind,
            central=g["point"].median(),
            scenario_min=g["point"].min(), scenario_max=g["point"].max(),
            full_lo=g["lo"].min(), full_hi=g["hi"].max(),
            s1_central=g[g["scenario"] == "S1_cohort_age"]["point"].median(),
            s2_central=g[g["scenario"] == "S2_cohort_period"]["point"].median(),
            s3_central=g[g["scenario"] == "S3_trend"]["point"].median(),
            capi_central=g[g["variant"] == "capi2017"]["point"].median(),
            all_central=g[g["variant"] == "all2017"]["point"].median()))
    return pd.DataFrame(out)


def summarize_backcast(bc):
    out = []
    for (yr, ind), g in bc.groupby(["target_year", "indicator"]):
        pts = g["point"]
        actual = g["actual_all"].iloc[0]
        row = dict(target_year=yr, indicator=ind, actual=actual,
                   locf=g["locf"].iloc[0],
                   scenario_min=pts.min(), scenario_max=pts.max(),
                   in_scenario_range=bool(pts.min() <= actual <= pts.max()),
                   in_full_range=bool(g["lo"].min() <= actual <= g["hi"].max()),
                   err_locf_pp=100 * (g["locf"].iloc[0] - actual))
        for scen in SCEN.values():
            s = g[g["scenario"] == scen]["point"]
            row["err_" + scen + "_pp"] = 100 * (s.mean() - actual)
        if yr == 2017:
            row["actual_capi"] = g["actual_capi"].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def chart(df, fs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping chart (pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ind in zip(axes, IND_PRIMARY):
        yrs = sorted(df["year"].unique())
        ax.plot(yrs, [100 * wshare(df[df["year"] == y], ind) for y in yrs],
                "o-", color="black", label="EVS, all respondents")
        capi = df[(df["year"] == 2017) & (df["mode"] == 1)]
        ax.plot([2017], [100 * wshare(capi, ind)], "s", color="grey",
                label="2017, face-to-face only")
        r = fs[fs["indicator"] == ind].iloc[0]
        ax.vlines(2036, 100 * r["full_lo"], 100 * r["full_hi"], color="tab:red",
                  lw=2, alpha=0.4, label="2036 full range")
        ax.vlines(2036, 100 * r["scenario_min"], 100 * r["scenario_max"],
                  color="tab:red", lw=6, label="2036 scenario range")
        ax.plot([2036], [100 * r["central"]], "D", color="tab:red")
        ax.set_title(ind)
        ax.set_ylabel("% of adults")
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("forecast_chart.png", dpi=150)
    print("saved forecast_chart.png")


def main(dta, xlsx):
    rng = np.random.default_rng(SEED)
    df = load(dta)
    pop = population_2036(xlsx)

    bc = run_backcast(df, rng)
    bc.to_csv("backcast.csv", index=False)
    bs = summarize_backcast(bc)
    bs.to_csv("backcast_summary.csv", index=False)

    fc = run_forecast(df, pop, rng)
    fc.to_csv("forecast_2036_all.csv", index=False)
    fs = summarize_forecast(fc)
    fs.to_csv("forecast_2036_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nBACK-CAST (percentage points, prediction minus actual):")
    cols = ["target_year", "indicator", "actual", "err_locf_pp",
            "err_S1_cohort_age_pp", "err_S2_cohort_period_pp",
            "err_S3_trend_pp", "in_scenario_range", "in_full_range"]
    print(bs[cols].round(3).to_string(index=False))
    print("\nFORECAST 2036 (share of adults):")
    print(fs[["indicator", "central", "scenario_min", "scenario_max",
              "full_lo", "full_hi"]].round(3).to_string(index=False))
    chart(df, fs)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python forecast_dk.py data.dta Denmark-2036.xlsx")
    main(sys.argv[1], sys.argv[2])
