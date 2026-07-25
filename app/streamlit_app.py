"""
Stage 5: Streamlit frontend.

Deliberately a thin client - it does zero modeling logic itself. It just
collects form input, POSTs to the FastAPI backend, and renders whatever
comes back. That separation is the whole point of splitting this into two
services (two HF Spaces) rather than one monolith: the API can be tested,
versioned, and even swapped out independently of the UI, and a real
production setup could put a mobile app or a different frontend in front
of the exact same API without touching a line of model code.

API_URL is read from an environment variable rather than hardcoded, since
this same file needs to point at localhost during local development and
at a different Space's URL once deployed to Hugging Face - hardcoding it
would mean editing code for every environment, which is exactly the kind
of thing config should handle instead.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Scoring", page_icon="\U0001F4B3", layout="wide")

st.title("Credit Risk Scoring")
st.caption(
    "Enter applicant details to get a default-risk score and a plain-language "
    "explanation of what's driving it. Backed by a LightGBM model trained on "
    "the Home Credit Default Risk dataset."
)

# --- a couple of preset profiles, purely so a demo (or an interviewer)
# doesn't have to type 20 fields by hand to see the app do something ---
PRESETS = {
    "-- fill in manually --": None,
    "Low-risk example": dict(
        income_total=180000, credit_amount=900000, annuity=45000, goods_price=850000,
        gender="F", age_years=34, own_car=False, own_realty=True, children_count=1,
        family_members=3, family_status="Married", education="Higher education",
        housing_type="House / apartment", income_type="Working",
        occupation_type="Core staff", years_employed=6, ext_source_2=0.62, ext_source_3=0.55,
    ),
    "Higher-risk example": dict(
        income_total=90000, credit_amount=1500000, annuity=None, goods_price=None,
        gender="M", age_years=23, own_car=False, own_realty=False, children_count=2,
        family_members=4, family_status="Single / not married", education="Lower secondary",
        housing_type="Rented apartment", income_type="Unemployed",
        occupation_type=None, years_employed=0, ext_source_2=None, ext_source_3=None,
    ),
}

preset_choice = st.selectbox("Quick-fill a sample profile (optional)", list(PRESETS.keys()))
preset = PRESETS[preset_choice] or {}

st.divider()

with st.form("applicant_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Loan details")
        contract_options = ["Cash loans", "Revolving loans"]
        contract_type = st.selectbox(
            "Contract type", contract_options,
            index=contract_options.index(preset["contract_type"]) if preset.get("contract_type") in contract_options else 0,
        )
        income_total = st.number_input(
            "Annual income", min_value=1.0, value=float(preset.get("income_total", 150000)), step=10000.0
        )
        credit_amount = st.number_input(
            "Requested credit amount", min_value=1.0, value=float(preset.get("credit_amount", 600000)), step=10000.0
        )
        annuity = st.number_input(
            "Loan annuity (monthly payment) - leave 0 to let the model estimate it",
            min_value=0.0, value=float(preset.get("annuity") or 0), step=1000.0
        )
        goods_price = st.number_input(
            "Price of goods the loan is for - leave 0 if not applicable",
            min_value=0.0, value=float(preset.get("goods_price") or 0), step=10000.0
        )

    with col2:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["F", "M"], index=0 if preset.get("gender", "F") == "F" else 1)
        age_years = st.number_input("Age", min_value=18, max_value=100, value=int(preset.get("age_years", 35)))
        own_car = st.checkbox("Owns a car", value=preset.get("own_car", False))
        car_age_years = st.number_input(
            "Car age (years) - only if car owner", min_value=0.0,
            value=float(preset.get("car_age_years") or 0), step=1.0
        )
        own_realty = st.checkbox("Owns property", value=preset.get("own_realty", False))
        children_count = st.number_input("Number of children", min_value=0, value=int(preset.get("children_count", 0)))
        family_members = st.number_input(
            "Family members (incl. applicant)", min_value=1, value=int(preset.get("family_members", 1))
        )
        family_status_options = ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"]
        family_status = st.selectbox(
            "Family status", family_status_options,
            index=family_status_options.index(preset["family_status"]) if preset.get("family_status") in family_status_options else 0,
        )
        education_options = ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
        education = st.selectbox(
            "Education", education_options,
            index=education_options.index(preset["education"]) if preset.get("education") in education_options else 0,
        )
        housing_type_options = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"]
        housing_type = st.selectbox(
            "Housing type", housing_type_options,
            index=housing_type_options.index(preset["housing_type"]) if preset.get("housing_type") in housing_type_options else 0,
        )

    with col3:
        st.subheader("Employment & bureau data")
        income_type_options = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
        income_type = st.selectbox(
            "Income type", income_type_options,
            index=income_type_options.index(preset["income_type"]) if preset.get("income_type") in income_type_options else 0,
        )
        occupation_type = st.text_input(
            "Occupation (optional, e.g. 'Core staff', 'Laborers')", value=preset.get("occupation_type") or ""
        )
        years_employed = st.number_input(
            "Years at current job (0 if unemployed/retired)", min_value=0.0,
            value=float(preset.get("years_employed") or 0), step=1.0
        )
        st.markdown("**External bureau scores** (0-1, optional)")
        st.caption(
            "These are normalized credit-bureau risk scores, similar in spirit to a CIBIL "
            "score - a walk-in applicant typically wouldn't know these themselves. Leave "
            "blank if unavailable; the model treats a missing bureau score as a signal in "
            "its own right (often indicating a thin credit file)."
        )
        ext_source_1 = st.number_input(
            "Bureau score 1 (0 = unknown)", min_value=0.0, max_value=1.0,
            value=float(preset.get("ext_source_1") or 0), step=0.01
        )
        ext_source_2 = st.number_input(
            "Bureau score 2 (0 = unknown)", min_value=0.0, max_value=1.0,
            value=float(preset.get("ext_source_2") or 0), step=0.01
        )
        ext_source_3 = st.number_input(
            "Bureau score 3 (0 = unknown)", min_value=0.0, max_value=1.0,
            value=float(preset.get("ext_source_3") or 0), step=0.01
        )

    submitted = st.form_submit_button("Score this applicant", use_container_width=True)


def _none_if_zero(x):
    """The form uses 0 as the 'not provided' sentinel for optional numeric
    fields (Streamlit number_input can't be left truly blank), so this
    converts that back to None before it hits the API, which is what
    actually tells the pipeline to impute rather than treat 0 as a real
    value - a real bureau score of exactly 0.0 would be a strange edge case
    we're accepting we can't distinguish from "not provided" here."""
    return None if x == 0 else x


if submitted:
    payload = {
        "contract_type": contract_type,
        "income_total": income_total,
        "credit_amount": credit_amount,
        "annuity": _none_if_zero(annuity),
        "goods_price": _none_if_zero(goods_price),
        "gender": gender,
        "age_years": age_years,
        "own_car": own_car,
        "car_age_years": _none_if_zero(car_age_years) if own_car else None,
        "own_realty": own_realty,
        "children_count": children_count,
        "family_members": family_members,
        "family_status": family_status,
        "education": education,
        "housing_type": housing_type,
        "income_type": income_type,
        "occupation_type": occupation_type or None,
        "years_employed": years_employed,
        "ext_source_1": _none_if_zero(ext_source_1),
        "ext_source_2": _none_if_zero(ext_source_2),
        "ext_source_3": _none_if_zero(ext_source_3),
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Couldn't reach the API at {API_URL}. Is the FastAPI server running? "
            f"Locally, start it with: `uvicorn api.main:app --reload --port 8000`"
        )
        st.stop()

    if resp.status_code != 200:
        st.error(f"API returned an error ({resp.status_code}): {resp.text}")
        st.stop()

    result = resp.json()

    st.divider()
    st.subheader("Result")

    band_colors = {"Low": "green", "Moderate": "orange", "High": "red", "Very High": "red"}
    band = result["risk_band"]

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted default probability", f"{result['default_probability']:.1%}")
    r2.markdown(f"**Risk band:** :{band_colors.get(band, 'gray')}[{band}]")
    decision_icon = "✅" if result["recommended_decision"] == "Approve" else "⛔"
    r3.markdown(f"**Recommendation:** {decision_icon} {result['recommended_decision']}")

    st.caption(
        f"Decision threshold used: {result['decision_threshold_used']:.3f} "
        f"(cost-optimized on the test set - see the business memo in the README for how this was derived, "
        f"not the conventional 0.5 default)"
    )

    st.markdown("#### What's driving this score")
    st.caption(
        "SHAP values are additive on the log-odds scale, not the probability scale directly - "
        "the numbers below are an approximate probability-point impact for readability, "
        "computed by asking 'how much would the prediction shift if this one factor were removed.'"
    )

    factors = result["top_factors"]
    for f in factors:
        impact = f["approx_probability_impact_pct"]
        direction_label = "increases risk" if f["direction"] == "increases_risk" else "decreases risk"
        bar_color = "#c53030" if f["direction"] == "increases_risk" else "#2f855a"
        # a simple manual bar via markdown/HTML - avoids pulling in a full
        # charting library just to draw eight horizontal bars
        max_width = max(abs(x["approx_probability_impact_pct"]) for x in factors) or 1
        width_pct = min(100, abs(impact) / max_width * 100)
        # administrative fields the form doesn't collect (FLAG_DOCUMENT_3,
        # ORGANIZATION_TYPE, etc.) come back as null - "not provided" reads
        # far better in the UI than the literal string "None"
        display_value = f["value"] if f["value"] is not None else "not provided"
        st.markdown(
            f"**{f['feature']}** = `{display_value}` &nbsp; "
            f"<span style='color:{bar_color}'>{direction_label} (~{abs(impact):.2f} pts)</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:{bar_color}; width:{width_pct}%; height:8px; border-radius:4px; margin-bottom:10px;'></div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    f"Connected to API at `{API_URL}`. This is a portfolio project, not a real lending "
    f"decision system - see the README for the business-tradeoff writeup and limitations."
)
