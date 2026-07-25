"""
This script exists purely so we have something to develop and test the
pipeline against locally. It is NOT a substitute for the real dataset.

Swap it out with the real thing like this:
    kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw/
    unzip data/raw/application_train.csv.zip -d data/raw/

The synthetic rows below mimic the actual application_train.csv schema
(same columns, same rough distributions, same missingness patterns) so the
cleaning/feature engineering code doesn't need to change when you drop the
real file in. I picked the columns that matter most for the story we're
telling (income, credit amount, employment, demographics, a few external
scores) rather than replicating all 122 - the real file has more, and the
pipeline just ignores whatever it doesn't recognise.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

# default rate in the real dataset is roughly 8% - keeping that same imbalance
# here so our imbalance-handling code gets a realistic workout
target = rng.choice([0, 1], size=n, p=[0.92, 0.08])

df = pd.DataFrame({
    "SK_ID_CURR": np.arange(100001, 100001 + n),
    "TARGET": target,
    "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
    "CODE_GENDER": rng.choice(["M", "F"], n, p=[0.34, 0.66]),
    "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.34, 0.66]),
    "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.7, 0.3]),
    "CNT_CHILDREN": rng.poisson(0.4, n),
    "AMT_INCOME_TOTAL": np.round(rng.lognormal(mean=11.8, sigma=0.5, size=n), -2),
    "AMT_CREDIT": np.round(rng.lognormal(mean=12.9, sigma=0.6, size=n), -2),
    "AMT_ANNUITY": np.round(rng.lognormal(mean=10, sigma=0.4, size=n), -1),
    "AMT_GOODS_PRICE": np.round(rng.lognormal(mean=12.8, sigma=0.6, size=n), -2),
    "NAME_INCOME_TYPE": rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        n, p=[0.51, 0.23, 0.18, 0.07, 0.01]
    ),
    "NAME_EDUCATION_TYPE": rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
        n, p=[0.71, 0.24, 0.03, 0.02]
    ),
    "NAME_FAMILY_STATUS": rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
        n, p=[0.64, 0.15, 0.1, 0.06, 0.05]
    ),
    "NAME_HOUSING_TYPE": rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
        n, p=[0.88, 0.05, 0.04, 0.03]
    ),
    # DAYS_BIRTH is negative days-from-today in the real data (weird but that's Home Credit for you)
    "DAYS_BIRTH": -rng.integers(21 * 365, 65 * 365, n),
    # DAYS_EMPLOYED has a famous data quality quirk in the real set: unemployed/retired folks
    # get a placeholder value of 365243 instead of a real number. Reproducing that on purpose
    # because handling it correctly is a real talking point.
    "DAYS_EMPLOYED": np.where(
        rng.random(n) < 0.18,
        365243,
        -rng.integers(0, 40 * 365, n)
    ),
    "OCCUPATION_TYPE": rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", None],
        n, p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15]
    ),
    "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
    "EXT_SOURCE_1": np.where(rng.random(n) < 0.56, np.nan, rng.beta(2, 2, n)),  # ~56% missing in real data
    "EXT_SOURCE_2": np.where(rng.random(n) < 0.002, np.nan, rng.beta(2, 2, n)),
    "EXT_SOURCE_3": np.where(rng.random(n) < 0.19, np.nan, rng.beta(2, 2, n)),
    "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.7, 0.2]),
    "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
})

# introduce a bit more realistic missingness in AMT_GOODS_PRICE and AMT_ANNUITY,
# since in real Home Credit data these have a small fraction of nulls
mask = rng.random(n) < 0.005
df.loc[mask, "AMT_GOODS_PRICE"] = np.nan
mask = rng.random(n) < 0.003
df.loc[mask, "AMT_ANNUITY"] = np.nan

df.to_csv("/home/claude/credit-risk-app/data/raw/application_train.csv", index=False)
print(f"wrote {len(df)} rows to data/raw/application_train.csv")
print(df.isnull().mean().sort_values(ascending=False).head(10))
