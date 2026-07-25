---
title: Credit Risk Scoring API
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Credit Risk Scoring API

FastAPI backend serving a LightGBM credit default risk model, with
per-prediction SHAP explanations. Trained on the Home Credit Default Risk
dataset (Kaggle).

- `POST /predict` — takes applicant details, returns a default probability,
  risk band, approve/reject recommendation, and the top SHAP factors
  driving that specific prediction.
- `GET /health` — basic liveness check.
- `GET /docs` — interactive API documentation (Swagger UI).

This Space is the backend for a paired Streamlit frontend Space. See the
full project (data pipeline, training, evaluation, business writeup) at:
[GitHub link here]
