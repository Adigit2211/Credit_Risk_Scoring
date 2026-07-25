"""
Using FastAPI's TestClient rather than a live curl script so this can run
in CI without needing a server actually listening on a port. TestClient
runs the app in-process, including the startup event that loads the model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

# TestClient only triggers startup/shutdown lifespan events (which is where
# the model actually gets loaded) when used as a context manager - using
# it as a bare object silently skips startup, and every request would 503.
client = TestClient(app)
client.__enter__()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_valid_probability():
    payload = {
        "income_total": 180000,
        "credit_amount": 900000,
        "annuity": 45000,
        "goods_price": 850000,
        "gender": "F",
        "age_years": 34,
        "own_car": False,
        "own_realty": True,
        "children_count": 1,
        "family_members": 3,
        "family_status": "Married",
        "education": "Higher education",
        "housing_type": "House / apartment",
        "income_type": "Working",
        "occupation_type": "Core staff",
        "years_employed": 6,
        "ext_source_2": 0.62,
        "ext_source_3": 0.55,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Moderate", "High", "Very High"}
    assert body["recommended_decision"] in {"Approve", "Reject / manual review"}
    assert len(body["top_factors"]) == 8


def test_predict_with_minimal_fields_still_works():
    """Only the strictly required fields (income, credit amount, gender,
    age) - everything else should fall back to imputed/missing gracefully
    rather than the request blowing up."""
    payload = {
        "income_total": 200000,
        "credit_amount": 500000,
        "gender": "M",
        "age_years": 40,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_invalid_gender_rejected():
    payload = {
        "income_total": 200000,
        "credit_amount": 500000,
        "gender": "X",
        "age_years": 40,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # pydantic validation error


def test_negative_income_rejected():
    payload = {
        "income_total": -5000,
        "credit_amount": 500000,
        "gender": "F",
        "age_years": 40,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


if __name__ == "__main__":
    test_health_check()
    test_predict_returns_valid_probability()
    test_predict_with_minimal_fields_still_works()
    test_invalid_gender_rejected()
    test_negative_income_rejected()
    print("all API tests passed")
