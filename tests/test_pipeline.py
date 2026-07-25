"""
These aren't exhaustive unit tests - they check the specific edge cases
that matter for this dataset: the DAYS_EMPLOYED placeholder trap, the
EXT_SOURCE missingness flags, and that no nulls leak through to the
model-ready output. If someone asks "how do you know your cleaning logic
is correct" in an interview, this file is the answer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.data_pipeline import fix_days_employed, clean, engineer_features, EMPLOYED_PLACEHOLDER


def test_days_employed_placeholder_gets_flagged_and_nulled():
    df = pd.DataFrame({"DAYS_EMPLOYED": [-500, EMPLOYED_PLACEHOLDER, -1200]})
    out = fix_days_employed(df)
    assert out["FLAG_NOT_EMPLOYED"].tolist() == [0, 1, 0]
    assert pd.isna(out.loc[1, "DAYS_EMPLOYED"])
    assert out.loc[0, "DAYS_EMPLOYED"] == -500  # untouched


def test_ext_source_missingness_flag_set_before_imputing():
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5, np.nan, 0.8],
        "EXT_SOURCE_2": [0.3, 0.4, np.nan],
        "EXT_SOURCE_3": [np.nan, np.nan, 0.6],
        "DAYS_EMPLOYED": [-100, -200, -300],
        "OCCUPATION_TYPE": ["Laborers", None, "Drivers"],
        "AMT_GOODS_PRICE": [1000, 2000, 3000],
        "AMT_ANNUITY": [100, 200, 300],
        "CNT_FAM_MEMBERS": [2, np.nan, 1],
    })
    out, _ = clean(df)
    assert out["EXT_SOURCE_1_MISSING"].tolist() == [0, 1, 0]
    # after imputation there should be no NaNs left in the EXT_SOURCE columns
    assert out["EXT_SOURCE_1"].isnull().sum() == 0
    assert out["EXT_SOURCE_3"].isnull().sum() == 0


def test_occupation_missing_becomes_own_category_not_mode():
    df = pd.DataFrame({
        "OCCUPATION_TYPE": ["Laborers", "Laborers", None],
        "EXT_SOURCE_1": [0.5, 0.5, 0.5],
        "EXT_SOURCE_2": [0.5, 0.5, 0.5],
        "EXT_SOURCE_3": [0.5, 0.5, 0.5],
        "DAYS_EMPLOYED": [-100, -100, -100],
        "AMT_GOODS_PRICE": [1000, 1000, 1000],
        "AMT_ANNUITY": [100, 100, 100],
        "CNT_FAM_MEMBERS": [1, 1, 1],
    })
    out, _ = clean(df)
    # should NOT have been silently turned into "Laborers" (the mode)
    assert out.loc[2, "OCCUPATION_TYPE"] == "Not_Specified"


def test_no_nulls_survive_full_clean_step():
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5, np.nan],
        "EXT_SOURCE_2": [np.nan, 0.4],
        "EXT_SOURCE_3": [0.6, np.nan],
        "DAYS_EMPLOYED": [EMPLOYED_PLACEHOLDER, -500],
        "OCCUPATION_TYPE": [None, "Drivers"],
        "AMT_GOODS_PRICE": [1000, np.nan],
        "AMT_ANNUITY": [np.nan, 200],
        "CNT_FAM_MEMBERS": [np.nan, 3],
    })
    out, _ = clean(df)
    assert out.isnull().sum().sum() == 0


def test_engineered_ratios_are_sane():
    df = pd.DataFrame({
        "AMT_CREDIT": [100000.0],
        "AMT_INCOME_TOTAL": [50000.0],
        "AMT_ANNUITY": [10000.0],
        "AMT_GOODS_PRICE": [90000.0],
        "DAYS_BIRTH": [-365 * 30],       # 30 years old
        "DAYS_EMPLOYED": [-365 * 5],     # 5 years employed
        "CNT_FAM_MEMBERS": [2.0],
    })
    out = engineer_features(df)
    assert out.loc[0, "CREDIT_TO_INCOME_RATIO"] == 2.0
    assert round(out.loc[0, "AGE_YEARS"], 1) == 30.0
    assert round(out.loc[0, "YEARS_EMPLOYED"], 1) == 5.0


if __name__ == "__main__":
    test_days_employed_placeholder_gets_flagged_and_nulled()
    test_ext_source_missingness_flag_set_before_imputing()
    test_occupation_missing_becomes_own_category_not_mode()
    test_no_nulls_survive_full_clean_step()
    test_engineered_ratios_are_sane()
    print("all pipeline tests passed")
