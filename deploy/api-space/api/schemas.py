"""
Request/response schemas for the credit risk API.

The field list here is deliberately smaller than the 87 columns the model
was trained on. That's a conscious choice, not an oversight - the training
data includes things like FLAG_DOCUMENT_3 (did the applicant submit
document type 3) or REG_CITY_NOT_WORK_CITY (does their registered city
differ from their work city) that come from internal bank systems, not
from anything an applicant would type into a form. Asking for them here
would just mean asking the user to guess at data they don't have.

Those fields aren't dropped from the model - they're still part of what
LightGBM was trained on, and it uses its native missing-value handling
for them when the live app doesn't supply them (see api/main.py). The
honest way to describe this in an interview is: "the deployed live-scoring
path uses fewer fields than the model was trained on, by design - the ones
omitted are administrative/internal-system fields with genuinely low
individual signal, and a production version would either wire those up
via internal system integrations or retrain a model on exactly the field
set the live form can supply." That second option - retraining a 'lean'
model on only the fields realistically available at application time -
avoids train-serve skew altogether and is what I'd actually recommend for
a real deployment; I've kept it out of scope here to keep the project
focused, but it's worth naming as the "next step" if asked.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ApplicantInput(BaseModel):
    # --- loan details ---
    contract_type: str = Field("Cash loans", description="Cash loans or Revolving loans")
    income_total: float = Field(..., gt=0, description="Annual income")
    credit_amount: float = Field(..., gt=0, description="Requested credit/loan amount")
    annuity: Optional[float] = Field(None, gt=0, description="Loan annuity (monthly payment). Leave blank to estimate.")
    goods_price: Optional[float] = Field(None, gt=0, description="Price of the goods the loan is for, if applicable")

    # --- demographics ---
    gender: str = Field(..., description="M or F")
    age_years: int = Field(..., ge=18, le=100)
    own_car: bool = False
    car_age_years: Optional[float] = Field(None, ge=0, description="Only relevant if own_car is true")
    own_realty: bool = False
    children_count: int = Field(0, ge=0)
    family_members: int = Field(1, ge=1)
    family_status: str = "Married"
    education: str = "Secondary / secondary special"
    housing_type: str = "House / apartment"

    # --- employment ---
    income_type: str = "Working"
    occupation_type: Optional[str] = None
    years_employed: Optional[float] = Field(
        None, description="Years at current job. Leave blank / 0 if unemployed or retired."
    )

    # --- external bureau scores (0-1 normalized), optional since a walk-in
    # applicant typically doesn't know these - they're the kind of thing a
    # loan officer would pull from a bureau integration, not ask the person
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        v = v.upper().strip()
        if v not in {"M", "F"}:
            raise ValueError("gender must be 'M' or 'F'")
        return v


class RiskFactor(BaseModel):
    feature: str
    value: str | float | int | None
    approx_probability_impact_pct: float
    direction: str


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    recommended_decision: str
    decision_threshold_used: float
    top_factors: list[RiskFactor]
