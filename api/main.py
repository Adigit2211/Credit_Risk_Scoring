"""
Stage 4: FastAPI backend.

One design decision worth explaining up front: this API does its
preprocessing (clean + engineer_features) *inside* the request handler,
using the exact same functions from src/data_pipeline.py that trained the
model. That's not an accident - it's the whole point of having refactored
clean() to accept saved imputation stats back in Stage 2. If the API had
its own separate, hand-rolled preprocessing logic, it would eventually
drift from what the model was actually trained on (someone updates
data_pipeline.py for the next training run, forgets the API has a copy of
the old logic, and now predictions are silently wrong). Importing the
same functions guarantees that can't happen.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ApplicantInput, PredictionResponse, RiskFactor
from src.data_pipeline import run_pipeline
from src.explain import RiskExplainer

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of loan default with a per-prediction SHAP explanation.",
    version="1.0.0",
)

# the Streamlit frontend runs on a different origin (a separate HF Space),
# so CORS needs to be open for it to call this API from the browser.
# Wide open ("*") is fine for a portfolio demo; a real deployment would
# restrict this to the frontend's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded once at startup, not per-request - joblib.load and rebuilding the
# SHAP TreeExplainer both cost real time, and every request hitting the
# same in-memory objects is the entire point of running this as a
# long-lived service instead of a script
_explainer: RiskExplainer | None = None
_feature_columns: list[str] | None = None
_imputation_stats: dict | None = None
_optimal_threshold: dict | None = None


@app.on_event("startup")
def load_artifacts():
    global _explainer, _feature_columns, _imputation_stats, _optimal_threshold
    _explainer = RiskExplainer.load("models")
    with open("models/feature_columns.json") as f:
        _feature_columns = json.load(f)
    with open("models/imputation_stats.json") as f:
        _imputation_stats = json.load(f)
    with open("models/optimal_threshold.json") as f:
        _optimal_threshold = json.load(f)


@app.get("/")
def root():
    return {
        "service": "Credit Risk Scoring API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _explainer is not None}


def _build_raw_row(applicant: ApplicantInput) -> pd.DataFrame:
    """Maps the form's friendly field names onto the raw Home Credit schema
    the pipeline expects, and does the couple of unit conversions
    (years -> the dataset's negative-days-from-today convention) needed to
    match what the model was trained on. Anything not set here is left out
    entirely - see api/schemas.py's docstring for why that's fine.
    """
    row = {
        "NAME_CONTRACT_TYPE": applicant.contract_type,
        "CODE_GENDER": applicant.gender,
        "FLAG_OWN_CAR": "Y" if applicant.own_car else "N",
        "FLAG_OWN_REALTY": "Y" if applicant.own_realty else "N",
        "CNT_CHILDREN": applicant.children_count,
        "AMT_INCOME_TOTAL": applicant.income_total,
        "AMT_CREDIT": applicant.credit_amount,
        # annuity/goods_price left as NaN if not supplied - clean() imputes
        # these with the saved training median, same as it would for any
        # batch row with a genuine data gap. Using np.nan (not python None)
        # matters here: pandas keeps a proper float64 dtype for np.nan, but
        # a bare None in a DataFrame constructor produces an 'object' dtype
        # column that LightGBM's predict() flatly refuses to accept, even
        # after it's been filled in by clean().
        "AMT_ANNUITY": applicant.annuity if applicant.annuity is not None else np.nan,
        "AMT_GOODS_PRICE": applicant.goods_price if applicant.goods_price is not None else np.nan,
        "NAME_INCOME_TYPE": applicant.income_type,
        "NAME_EDUCATION_TYPE": applicant.education,
        "NAME_FAMILY_STATUS": applicant.family_status,
        "NAME_HOUSING_TYPE": applicant.housing_type,
        "DAYS_BIRTH": -int(applicant.age_years * 365.25),
        # the dataset's own convention: unemployed/retired gets the 365243
        # placeholder rather than a real days-employed figure. fix_days_employed()
        # in the pipeline already knows how to handle that value correctly.
        "DAYS_EMPLOYED": (
            365243 if not applicant.years_employed
            else -int(applicant.years_employed * 365.25)
        ),
        "OCCUPATION_TYPE": applicant.occupation_type,  # fine as None - it's a categorical column
        "CNT_FAM_MEMBERS": applicant.family_members,
        "OWN_CAR_AGE": applicant.car_age_years if applicant.car_age_years is not None else np.nan,
        "EXT_SOURCE_1": applicant.ext_source_1 if applicant.ext_source_1 is not None else np.nan,
        "EXT_SOURCE_2": applicant.ext_source_2 if applicant.ext_source_2 is not None else np.nan,
        "EXT_SOURCE_3": applicant.ext_source_3 if applicant.ext_source_3 is not None else np.nan,
    }
    return pd.DataFrame([row])


def _risk_band(probability: float) -> str:
    """Simple banding for the UI - not the actual decision (that's the
    threshold below), just a human-readable label."""
    if probability < 0.05:
        return "Low"
    elif probability < 0.15:
        return "Moderate"
    elif probability < 0.30:
        return "High"
    return "Very High"


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantInput):
    if _explainer is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    raw_row = _build_raw_row(applicant)
    processed_row, _ = run_pipeline(raw_row, stats=_imputation_stats)

    # reindex to the exact column set/order the model trained on - any
    # administrative field the form doesn't collect (FLAG_DOCUMENT_3,
    # REGION_RATING_CLIENT, etc.) shows up here as a real NaN column,
    # which LightGBM handles natively via its learned missing-value routing
    processed_row = processed_row.reindex(columns=_feature_columns)

    result = _explainer.explain_one(processed_row, top_n=8)

    threshold = _optimal_threshold["optimal_threshold"]
    decision = "Reject / manual review" if result["default_probability"] >= threshold else "Approve"

    return PredictionResponse(
        default_probability=result["default_probability"],
        risk_band=_risk_band(result["default_probability"]),
        recommended_decision=decision,
        decision_threshold_used=threshold,
        top_factors=[
            RiskFactor(
                feature=f["feature"],
                value=f["value"],
                approx_probability_impact_pct=round(f["approx_probability_impact"] * 100, 3),
                direction=f["direction"],
            )
            for f in result["top_factors"]
        ],
    )
