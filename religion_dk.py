"""
Religion in Europe, 2036: step 1 (measurement system).

Computes believing, belonging and practice for Denmark by EVS wave and
birth cohort, from the EVS Trend File 1981-2017 (ZA7503, v3.0.0).

Run:  python religion_dk.py path/to/ZA7503_v3-0-0.dta
Output: results_by_wave_cohort.csv, results_by_wave.csv
The data file is NOT part of the repository (see .gitignore).

Definitions (value codes checked in the ZA7503 variable report):
  believing : F050 == 1            believes in God (primary)
              F063 >= 6            God important (scale 1-10)  [robustness]
              F034 == 1            "religious person"          [robustness]
  belonging : F025 > 0             belongs to a denomination (0 = none)
  practice  : F028 in 1..3         attends services at least monthly
              F028 in 1..2         attends at least weekly     [robustness]
Missing values in the file are negative codes (-1 to -5); they are dropped
separately for each indicator.
Weights: S017 (EVS weight). Unweighted n is reported next to each share.
"""
import sys
import numpy as np
import pandas as pd

COUNTRY = 208  # Denmark (ISO 3166-1 numeric, variable S003)
MIN_N = 30     # cells below this unweighted n are flagged as unreliable

COHORT_BINS = [-np.inf, 1939, 1949, 1959, 1969, 1979, np.inf]
COHORT_LABELS = ["<=1939", "1940-49", "1950-59", "1960-69", "1970-79", "1980+"]

NEEDED = ["S002EVS", "S003", "S020", "S017", "X002",
          "F050", "F063", "F034", "F025", "F028"]


def load(path):
    df = pd.read_stata(path, convert_categoricals=False, columns=NEEDED)
    df = df[df["S003"] == COUNTRY].copy()
    df = df.rename(columns={"S002EVS": "wave", "S020": "year", "S017": "w"})
    # X002 is -2 for some 2017 respondents (no answer); no cohort then
    df.loc[df["X002"] < 1800, "X002"] = np.nan
    df["cohort"] = pd.cut(df["X002"], COHORT_BINS, labels=COHORT_LABELS)
    return df


def indicators(df):
    """Return a 0/1/NaN column per indicator."""
    out = pd.DataFrame(index=df.index)

    def ind(var, cond):
        valid = df[var] >= 0
        return np.where(valid, cond.astype(float), np.nan)

    out["believing_god"] = ind("F050", df["F050"] == 1)
    out["believing_importance"] = ind("F063", df["F063"] >= 6)
    out["believing_religious_person"] = ind("F034", df["F034"] == 1)
    out["belonging"] = ind("F025", df["F025"] > 0)
    out["practice_monthly"] = ind("F028", df["F028"].between(1, 3))
    out["practice_weekly"] = ind("F028", df["F028"].between(1, 2))
    return out


def weighted_share(g, col):
    d = g[[col, "w"]].dropna()
    if len(d) == 0:
        return pd.Series({"share": np.nan, "n": 0})
    return pd.Series({"share": np.average(d[col], weights=d["w"]),
                      "n": len(d)})


def table(df, by):
    cols = [c for c in df.columns if c.startswith(("believing", "belonging", "practice"))]
    rows = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        for c in cols:
            r = weighted_share(g, c)
            rows.append(dict(zip(by, key), indicator=c,
                             share=r["share"], n=int(r["n"]),
                             reliable=bool(r["n"] >= MIN_N)))
    return pd.DataFrame(rows)


def main(path):
    df = load(path)
    df = pd.concat([df, indicators(df)], axis=1)

    by_wave = table(df, ["wave", "year"])
    by_cohort = table(df.dropna(subset=["cohort"]), ["wave", "year", "cohort"])

    by_wave.to_csv("results_by_wave.csv", index=False)
    by_cohort.to_csv("results_by_wave_cohort.csv", index=False)

    print("Respondents per wave:")
    print(df.groupby(["wave", "year"]).size().to_string())
    print("\nShares by wave (weighted):")
    print(by_wave.pivot(index="indicator", columns="year", values="share").round(3).to_string())
    print("\nCohort cells with n < %d: %d of %d" %
          (MIN_N, (~by_cohort["reliable"]).sum(), len(by_cohort)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python religion_dk.py path/to/ZA7503_v3-0-0.dta")
    main(sys.argv[1])
