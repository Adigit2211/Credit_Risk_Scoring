---
title: Credit Risk Scoring UI
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
---

# Credit Risk Scoring - UI

Streamlit frontend for the Credit Risk Scoring project. Enter applicant
details, get a default-risk score and a SHAP-based explanation of what's
driving it.

This is a thin client - all the modeling logic lives in a separate FastAPI
backend Space. Set the `API_URL` variable (below, or as a Space secret/
variable) to point at that backend's URL.

Full project (data pipeline, training, evaluation, business writeup) at:
[GitHub link here]
