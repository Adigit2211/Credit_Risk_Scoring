"""
Data cleaning and feature engineering for the Home Credit default risk dataset.

The guiding rule I've used throughout: the missing-value strategy depends on
*why* a value is missing, not just that it's missing. There are three
distinct mechanisms in this dataset and each needs a different fix:

  1. Structurally missing - the field doesn't apply to this person
     (e.g. DAYS_EMPLOYED for someone who's unemployed/retired). Filling
     this with a mean would be actively wrong; it needs a flag + sentinel.

  2. Missing at random / data collection gaps - things like AMT_GOODS_PRICE
     that are missing for a small, unremarkable fraction of rows. Simple
     imputation (median) is fine because there's no pattern to preserve.

  3. Missing because it's genuinely unknown but informative - EXT_SOURCE_1
     is missing for over half the applicants. That's suspiciously high for
     "random data entry issue" - it more likely means that particular
     external bureau simply doesn't have a score for certain applicant
     segments (e.g. thin-file / first-time borrowers), which is itself
     a risk signal. So I keep a missingness flag AND impute, letting the
     model use both.

This distinction is exactly the kind of thing that's worth walking an
interviewer through - "I used median imputation everywhere" is a weaker
answer than "I checked *why* each field was missing and treated three
different mechanisms differently."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# Home Credit's well-known data quirk: DAYS_EMPLOYED uses 365243 as a
# placeholder for "not employed" instead of NaN. If you don't catch this,
# it silently becomes the single biggest outlier in the dataset and wrecks
# any model that's sensitive to scale (logistic regression especially).
EMPLOYED_PLACEHOLDER = 365243

# The full application_train.csv has ~50 columns describing the applicant's
# building (APARTMENTS_AVG, YEARS_BUILD_MODE, WALLSMATERIAL_MODE, and their
# _MODE/_MEDI variants - three versions of basically the same building survey
# stat). I'm dropping this whole block, and it's worth being explicit about
# why rather than just quietly excluding them:
#   1. Most are 50-70% missing - by far the sparsest block in the dataset,
#      sparser than even EXT_SOURCE_1.
#   2. The three variants (_AVG/_MODE/_MEDI) of each stat are highly
#      collinear with each other, so keeping all three barely adds signal
#      over keeping one - and I'm not convinced keeping even one earns its
#      complexity budget here.
#   3. Practically: nobody applying for a loan through our app is going to
#      know their building's "common area mode" off the top of their head.
#      A live-inference form has to stay answerable, and these fields fail
#      that test hardest of anything in the dataset.
# For an interview, the honest framing is "individually low signal, painful
# to collect at inference time, so I traded a small amount of possible lift
# for a form a real applicant could actually fill in" - not "I didn't notice
# them."
BUILDING_QUALITY_COLS = [
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
    "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
    "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",
    "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
    "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI",
    "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
    "TOTALAREA_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]

# columns we're intentionally not using for the live-inference model,
# either because a walk-in applicant couldn't report them (DAYS_LAST_PHONE_CHANGE
# is odd to ask someone) or because they're IDs / leakage risks
DROP_COLS = ["SK_ID_CURR"] + BUILDING_QUALITY_COLS

TARGET_COL = "TARGET"


@dataclass
class FeatureLists:
    """Just a container so downstream code (train.py, api) can pull the same
    column groupings without re-deriving them by hand and risking drift."""
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    binary_flag: list[str] = field(default_factory=list)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the 365243 placeholder properly - flag it, then treat it as
    missing rather than as a real (huge, positive) number of days employed.
    """
    df = df.copy()
    df["FLAG_NOT_EMPLOYED"] = (df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER, "DAYS_EMPLOYED"] = np.nan
    return df


def clean(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Applies the per-feature-type missing value strategy described up top.

    `stats` controls where imputation values (medians etc.) come from:
      - None (training mode): compute them from `df` itself, and return
        them so the caller can save them.
      - a dict (inference mode): use the saved training-time values instead
        of recomputing from `df`. This matters a lot for single-row
        predictions in the API - you cannot take "the median" of one row,
        and even if you could, a fresh median per request would mean the
        same applicant gets a different imputed value depending on what
        else happened to be in the batch, which is not a property you want
        from a production scoring service.
    """
    df = df.copy()
    df = fix_days_employed(df)
    is_training = stats is None
    stats = {} if is_training else dict(stats)  # copy so we don't mutate the caller's dict

    # --- structurally missing: EXT_SOURCE_* scores ---
    # These are external credit bureau scores. When missing, it's rarely
    # random - it usually means that bureau had no file on the applicant.
    # We keep the "no score available" signal as its own flag, then impute
    # with the median so the model still has a usable number to work with.
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isnull().astype(int)
            if is_training:
                stats[f"{col}_median"] = float(df[col].median())
            # .astype(float) here isn't decorative - a single-row DataFrame
            # built from a dict with a mix of None and float values can end
            # up with 'object' dtype, and fillna() alone doesn't upgrade
            # that back to float64. LightGBM's predict() hard-rejects
            # object-dtype columns, so this has to be explicit.
            df[col] = df[col].fillna(stats[f"{col}_median"]).astype(float)

    # --- occupation: missing likely means self-employed / not formally
    # recorded, which is meaningfully different from any listed occupation.
    # Rather than impute with the mode (which would misrepresent these
    # people as "Laborers" or whatever's most common), we give missingness
    # its own category. This preserves the signal instead of erasing it.
    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Not_Specified")

    # --- garden-variety small-fraction numeric gaps: AMT_GOODS_PRICE,
    # AMT_ANNUITY. Under 1% missing, no discernible pattern - median
    # imputation is the pragmatic, defensible choice here. I wouldn't
    # bother building a flag column for a <1% missing rate; the extra
    # feature would mostly just be noise.
    for col in ["AMT_GOODS_PRICE", "AMT_ANNUITY"]:
        if col in df.columns:
            if is_training:
                stats[f"{col}_median"] = float(df[col].median())
            df[col] = df[col].fillna(stats[f"{col}_median"]).astype(float)

    # DAYS_EMPLOYED itself (after removing the placeholder) - impute with
    # median of the *actually employed* population, since that's the
    # population this column describes now
    if "DAYS_EMPLOYED" in df.columns:
        if is_training:
            stats["DAYS_EMPLOYED_median"] = float(df["DAYS_EMPLOYED"].median())
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(stats["DAYS_EMPLOYED_median"]).astype(float)

    # CNT_FAM_MEMBERS - a handful of nulls in the real dataset, default to 1
    # (a single applicant) rather than a fractional median, since this is a count
    if "CNT_FAM_MEMBERS" in df.columns:
        df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(1).astype(float)

    # --- OWN_CAR_AGE: same structural-missingness story as DAYS_EMPLOYED.
    # It's ~66% missing, and that lines up almost exactly with the fraction
    # of applicants who have FLAG_OWN_CAR == 'N' - i.e. it's not missing at
    # random, it's missing *because the field doesn't apply*. Filling with
    # the median car age would fabricate a car for people who don't have
    # one, so: flag it, then impute using only car-owners' median.
    if "OWN_CAR_AGE" in df.columns:
        df["FLAG_CAR_AGE_MISSING"] = df["OWN_CAR_AGE"].isnull().astype(int)
        if is_training:
            stats["OWN_CAR_AGE_owner_median"] = float(
                df.loc[df["OWN_CAR_AGE"].notnull(), "OWN_CAR_AGE"].median()
            )
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(stats["OWN_CAR_AGE_owner_median"]).astype(float)

    # --- AMT_REQ_CREDIT_BUREAU_* (6 cols): number of times credit bureau was
    # queried about this applicant over various windows (hour/day/week/month/
    # quarter/year). ~13.5% missing, and missing here most plausibly means
    # "no inquiry on record" rather than "unknown count" - so 0 is a
    # defensible fill, not just a lazy default. Treating this the same as
    # the EXT_SOURCE columns (impute with median) would overstate how often
    # quiet applicants get checked on.
    bureau_inquiry_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    for col in bureau_inquiry_cols:
        df[col] = df[col].fillna(0).astype(float)

    # --- NAME_TYPE_SUITE: who accompanied the applicant. Under 0.5% missing,
    # no strong pattern - "Not_Specified" as its own bucket instead of
    # guessing "Unaccompanied" for people we don't actually know were alone.
    if "NAME_TYPE_SUITE" in df.columns:
        df["NAME_TYPE_SUITE"] = df["NAME_TYPE_SUITE"].fillna("Not_Specified")

    # --- social circle observation/default counts and DAYS_LAST_PHONE_CHANGE:
    # trivial missing fractions (<1%), no evidence of a structural pattern,
    # plain median imputation is the pragmatic call here same as AMT_ANNUITY etc.
    trivial_numeric_gaps = [
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
    ]
    for col in trivial_numeric_gaps:
        if col in df.columns:
            if is_training:
                stats[f"{col}_median"] = float(df[col].median())
            df[col] = df[col].fillna(stats[f"{col}_median"]).astype(float)

    return df, stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven feature engineering. Each of these is something an
    actual credit analyst would look at - that's deliberate, since
    "I engineered features that reflect how underwriters actually think"
    is a much stronger interview line than "I did polynomial feature
    combos until CV score went up."
    """
    df = df.copy()

    # debt-to-income: how much of their income the requested credit represents.
    # this is *the* classic underwriting ratio
    df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    # annuity-to-income: what fraction of income goes to the loan payment
    # every month - closer to how repayment burden is actually felt
    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # what fraction of the credit is actually covering the goods price
    # (vs. fees, add-ons, insurance bundled into the loan) - a loan that's
    # much bigger than the goods price it's nominally for can be a red flag
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)

    # implied loan term in years, back-calculated from credit / annuity.
    # longer implied terms at a similar income level often correlate with
    # borrowers stretching themselves thin
    df["ANNUITY_LENGTH_YEARS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)

    # age in years - DAYS_BIRTH is negative days-from-today in the raw data,
    # converting to something human-readable also makes SHAP output legible
    # in the final app rather than showing "DAYS_BIRTH = -14235"
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25)

    # years employed, same logic
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365.25)

    # employment as a fraction of age - a 25-year-old with 10 years employed
    # is a very different story to a 45-year-old with 10 years employed,
    # this ratio captures career stability relative to life stage
    df["EMPLOYED_TO_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]

    # income per family member - raw income can be misleading for a family
    # of 5 vs. a single applicant with the same take-home pay
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    # average of the three external bureau scores - a simple ensemble of
    # external signals tends to be one of the single strongest predictors
    # in this dataset (well documented in the original Kaggle competition
    # discussions), worth engineering explicitly rather than hoping the
    # model finds the combination on its own
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

    return df


def prep_categoricals_for_lgbm(X: pd.DataFrame, flists, categories: dict | None = None):
    """LightGBM wants categorical columns as pandas 'category' dtype rather
    than one-hot. The tricky bit for a deployed model: at inference time a
    single new row must be cast against the *same* category set seen during
    training, or LightGBM will silently treat an unseen category as missing.
    So this function either derives the category list (training time) or
    applies a saved one (inference time) - same function, two modes, so
    train and serve can't quietly diverge.

    Lives here rather than in train.py deliberately - this function is on
    the serving path (the API calls it via explain.py on every prediction),
    and train.py imports things like imbalanced-learn that a lean
    production deployment has no business pulling in just to reach one
    small utility function.
    """
    X = X.copy()
    learned_categories = {}
    for col in flists.categorical:
        if categories is None:
            X[col] = X[col].astype("category")
            learned_categories[col] = X[col].cat.categories.tolist()
        else:
            X[col] = pd.Categorical(X[col], categories=categories[col])
    return X, (learned_categories if categories is None else categories)


def get_feature_lists(df: pd.DataFrame) -> FeatureLists:
    """Splits columns into numeric / categorical / binary-flag groups so
    train.py can build the right preprocessing for each without repeating
    this logic by hand."""
    exclude = {TARGET_COL, *DROP_COLS}
    numeric, categorical, binary = [], [], []

    for col in df.columns:
        if col in exclude:
            continue
        # not just `dtype == object` - newer pandas can give string columns
        # a native "str" dtype instead of "object", so check for either
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            categorical.append(col)
        elif set(df[col].dropna().unique()).issubset({0, 1}):
            binary.append(col)
        else:
            numeric.append(col)

    return FeatureLists(numeric=numeric, categorical=categorical, binary_flag=binary)


def run_pipeline(raw_path_or_df, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """End-to-end: load -> clean -> engineer. This is the single function
    both train.py and the API's preprocessing step call, so training and
    serving can never quietly drift apart from each other.

    Accepts either a CSV path (training) or an already-loaded DataFrame
    (the API passes a single-row DataFrame built from the request body).
    `stats=None` means "training mode, derive imputation values from this
    data and hand them back so they can be saved." `stats={...}` means
    "inference mode, use these previously-saved values" - required for
    single-row predictions where there's no meaningful median to compute.
    """
    df = load_raw(raw_path_or_df) if isinstance(raw_path_or_df, str) else raw_path_or_df.copy()
    df, stats = clean(df, stats=stats)
    df = engineer_features(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df, stats


if __name__ == "__main__":
    processed, imputation_stats = run_pipeline("data/raw/application_train.csv")
    processed.to_parquet("data/processed/train_processed.parquet", index=False)

    import json
    with open("models/imputation_stats.json", "w") as f:
        json.dump(imputation_stats, f, indent=2)

    flists = get_feature_lists(processed)
    print(f"rows: {len(processed)}, cols: {processed.shape[1]}")
    print(f"numeric: {len(flists.numeric)}, categorical: {len(flists.categorical)}, binary flags: {len(flists.binary_flag)}")
    print(f"remaining nulls:\n{processed.isnull().sum()[processed.isnull().sum() > 0]}")
    print(f"saved imputation stats to models/imputation_stats.json ({len(imputation_stats)} values)")
