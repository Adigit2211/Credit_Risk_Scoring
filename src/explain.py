"""
Stage 3: explainability.

Why SHAP and not, say, LightGBM's built-in feature_importances_: global
importance tells you "EXT_SOURCE_2 matters a lot across the whole
portfolio," which is true but useless to a loan officer looking at one
specific applicant. SHAP gives a *per-prediction* breakdown - for this
one person, which features pushed their risk score up and which pushed
it down, and by how much. That's what the app needs to show, and it's
also the more defensible answer if a rejected applicant ever asks "why."

Why TreeExplainer specifically: it's exact (not an approximation like
KernelExplainer) and fast for tree ensembles, because it can walk the
actual tree structure to compute Shapley values rather than sampling
coalitions of features like the model-agnostic explainers have to.
There's no reason to pay KernelExplainer's cost when the model is a
tree ensemble.

One thing worth being upfront about: SHAP values are additive relative
to a baseline (the model's average prediction over the training set),
not "this feature caused this outcome" in a causal sense. I make sure
the app copy reflects that distinction rather than overclaiming.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import shap

from src.data_pipeline import prep_categoricals_for_lgbm


class RiskExplainer:
    """Wraps a fitted LightGBM model + SHAP TreeExplainer so the API layer
    doesn't need to know anything about SHAP's internals - it just calls
    explain_one() and gets back a clean, JSON-serializable structure.
    """

    def __init__(self, model, flists, categories: dict):
        self.model = model
        self.flists = flists
        self.categories = categories
        # TreeExplainer needs the model, not raw data, to build itself -
        # this doesn't need a background dataset for tree models the way
        # KernelExplainer would
        self.explainer = shap.TreeExplainer(model)

    @classmethod
    def load(cls, models_dir: str = "models") -> "RiskExplainer":
        model = joblib.load(f"{models_dir}/lgbm_model.pkl")
        flists = joblib.load(f"{models_dir}/feature_lists.pkl")
        with open(f"{models_dir}/lgbm_categories.json") as f:
            categories = json.load(f)
        return cls(model, flists, categories)

    def explain_one(self, row: pd.DataFrame, top_n: int = 8) -> dict:
        """row: single-row DataFrame with the same columns the model was
        trained on (post cleaning/feature-engineering, pre categorical-cast -
        this function handles the cast itself using the saved categories).

        Returns a dict with the predicted probability, the SHAP base value
        (the model's average prediction absent any feature info), and the
        top contributing features sorted by |impact|, both positive
        (risk-increasing) and negative (risk-decreasing).
        """
        row_c, _ = prep_categoricals_for_lgbm(row, self.flists, categories=self.categories)
        proba = float(self.model.predict_proba(row_c)[0, 1])

        shap_values = self.explainer.shap_values(row_c)
        # binary classification with LightGBM's sklearn wrapper returns a
        # single array of shape (1, n_features) for the positive class
        # when using TreeExplainer in this mode - guard for the occasional
        # list-of-two-arrays shape some shap/lightgbm version combos return
        if isinstance(shap_values, list):
            values = shap_values[1][0]
            base_value = self.explainer.expected_value[1]
        else:
            values = shap_values[0]
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[-1]

        feature_names = row_c.columns.tolist()
        feature_values = row_c.iloc[0].to_dict()
        logit_total = base_value + values.sum()  # this recovers `proba` via sigmoid - sanity-checked in dev

        contributions = sorted(
            zip(feature_names, values, [feature_values[f] for f in feature_names]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        return {
            "default_probability": proba,
            "base_value_logodds": float(base_value),
            "top_factors": [
                {
                    "feature": name,
                    "value": _jsonify(val),
                    "shap_logodds": float(shap_val),
                    # SHAP itself is additive on the log-odds scale, not the
                    # probability scale - a raw log-odds number means little
                    # to someone reading the app, so we also report an
                    # approximate probability-point impact: "how much would
                    # the predicted probability shift if this one factor's
                    # contribution were removed, holding everything else
                    # fixed." It's a standard, order-independent way to make
                    # SHAP output human-readable; it's an approximation
                    # (probability isn't linear in log-odds) but it's honest
                    # about being one, and it's what actually gets shown
                    # in the UI rather than the raw log-odds number.
                    "approx_probability_impact": float(
                        proba - _sigmoid(logit_total - shap_val)
                    ),
                    "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                }
                for name, shap_val, val in contributions
            ],
        }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _jsonify(val):
    """SHAP/pandas hand back numpy scalars and Categorical values that
    json.dumps chokes on - flatten to plain python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if pd.isna(val):
        return None
    return str(val) if not isinstance(val, (int, float)) else val


if __name__ == "__main__":
    # quick smoke test: explain one row from the test set and print it,
    # just to confirm the whole chain works before wiring it into the API
    test_df = pd.read_parquet("data/processed/test_set.parquet")
    sample_row = test_df.drop(columns=["TARGET"]).iloc[[0]]

    explainer = RiskExplainer.load()
    result = explainer.explain_one(sample_row)

    print(f"predicted default probability: {result['default_probability']:.3f}")
    print(f"base rate (log-odds scale, model average): {result['base_value_logodds']:.3f}")
    print("\ntop contributing factors:")
    for f in result["top_factors"]:
        arrow = "^ risk" if f["direction"] == "increases_risk" else "v risk"
        pp = f["approx_probability_impact"] * 100
        print(f"  {f['feature']:<30} = {f['value']!s:<15} {arrow}  (~{pp:+.2f} pts)")
