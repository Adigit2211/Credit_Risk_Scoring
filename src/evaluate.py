"""
Stage 2 (continued): evaluation, framed in business terms rather than just
reporting PR-AUC and calling it done.

The core idea: 0.5 is not a meaningful decision threshold for this problem.
It's the threshold you'd pick if a false negative (approving a loan that
defaults) and a false positive (rejecting an applicant who would've repaid)
cost the business the same amount. They very much don't:

  - False negative cost: loss given default. If a loan defaults, the bank
    doesn't lose the full AMT_CREDIT - some of it is usually recovered
    through collections/collateral. I'm using a loss-given-default (LGD)
    assumption of 45%, which is a commonly cited unsecured-consumer-credit
    figure, applied to each applicant's actual requested credit amount
    rather than a single portfolio-wide number - so a ₹50L default costs
    more than a ₹2L default in this analysis, which is obviously true in
    reality and would be lost if I just used a flat cost per error.

  - False positive cost: opportunity cost of a lost customer - the interest
    margin the bank would have earned. I'm approximating this as a small
    percentage of the credit amount (assumed ~2.5% net interest margin over
    the loan), again per-applicant rather than flat.

Both LGD and margin are assumptions, not measured facts - I'm treating them
as adjustable parameters precisely because in an interview the honest
answer to "where did 45% come from" is "a reasonable industry figure I
picked, and here's what changes if you disagree" - not a number I'm
pretending is precise.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.data_pipeline import TARGET_COL
from src.data_pipeline import prep_categoricals_for_lgbm

# business assumptions - deliberately named and up top so they're easy to
# challenge/change, not buried inside a formula somewhere
LOSS_GIVEN_DEFAULT = 0.45      # fraction of credit amount lost if a loan defaults
NET_INTEREST_MARGIN = 0.025    # fraction of credit amount earned as profit if a good loan is approved


def compute_row_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant FN/FP costs, driven by their actual requested credit
    amount rather than a single flat number for the whole portfolio."""
    out = df.copy()
    out["fn_cost"] = out["AMT_CREDIT"] * LOSS_GIVEN_DEFAULT
    out["fp_cost"] = out["AMT_CREDIT"] * NET_INTEREST_MARGIN
    return out


def threshold_cost_sweep(y_true: np.ndarray, y_proba: np.ndarray, fn_cost: np.ndarray,
                          fp_cost: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """For each candidate threshold, total expected cost = sum of FN costs
    for defaults we missed + sum of FP costs for good applicants we
    rejected. We sweep thresholds and pick whichever minimizes this, rather
    than defaulting to 0.5.
    """
    if thresholds is None:
        # finer resolution near the low end - given the cost asymmetry here
        # (FN costs ~18x FP costs), the optimal threshold lands well below
        # the naive 0.5 and a coarse 0.01 step misses the actual minimum
        thresholds = np.concatenate([np.arange(0.005, 0.15, 0.0025), np.arange(0.15, 0.95, 0.01)])

    rows = []
    for t in thresholds:
        pred_default = (y_proba >= t).astype(int)
        false_negatives = (pred_default == 0) & (y_true == 1)   # missed a real default
        false_positives = (pred_default == 1) & (y_true == 0)   # rejected a good applicant

        total_cost = fn_cost[false_negatives].sum() + fp_cost[false_positives].sum()
        rows.append({
            "threshold": t,
            "total_cost": total_cost,
            "n_false_negatives": int(false_negatives.sum()),
            "n_false_positives": int(false_positives.sum()),
            "approval_rate": float((pred_default == 0).mean()),
        })
    return pd.DataFrame(rows)


def find_optimal_threshold(cost_df: pd.DataFrame) -> dict:
    best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    return best_row.to_dict()


if __name__ == "__main__":
    print("loading test set and model...")
    test_df = pd.read_parquet("data/processed/test_set.parquet")
    y_test = test_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL])

    model = joblib.load("models/lgbm_model.pkl")
    flists = joblib.load("models/feature_lists.pkl")
    with open("models/lgbm_categories.json") as f:
        categories = json.load(f)

    X_test_c, _ = prep_categoricals_for_lgbm(X_test, flists, categories=categories)
    y_proba = model.predict_proba(X_test_c)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"test PR-AUC: {pr_auc:.4f}")

    # what does precision/recall look like at the default 0.5 threshold,
    # just to have as a point of comparison for the cost-optimal one
    precision, recall, thresh = precision_recall_curve(y_test, y_proba)
    print(f"\n(reference) at threshold 0.5:")
    idx = np.argmin(np.abs(thresh - 0.5))
    print(f"  precision: {precision[idx]:.3f}, recall: {recall[idx]:.3f}")

    print("\n--- cost-based threshold sweep ---")
    costs = compute_row_costs(test_df)
    cost_df = threshold_cost_sweep(
        y_test, y_proba,
        fn_cost=costs["fn_cost"].values,
        fp_cost=costs["fp_cost"].values,
    )
    optimal = find_optimal_threshold(cost_df)
    print(f"cost-minimizing threshold: {optimal['threshold']:.2f}")
    print(f"  total expected cost at this threshold: {optimal['total_cost']:,.0f}")
    print(f"  false negatives (missed defaults): {optimal['n_false_negatives']}")
    print(f"  false positives (rejected good applicants): {optimal['n_false_positives']}")
    print(f"  approval rate: {optimal['approval_rate']:.1%}")

    # cost at the naive 0.5 threshold, for direct comparison
    naive_row = cost_df.iloc[(cost_df["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    savings = naive_row["total_cost"] - optimal["total_cost"]
    print(f"\ncost at naive 0.5 threshold: {naive_row['total_cost']:,.0f}")
    print(f"expected saving from cost-based threshold vs naive 0.5: {savings:,.0f} "
          f"({savings / naive_row['total_cost']:.1%} reduction)")

    cost_df.to_csv("models/threshold_cost_sweep.csv", index=False)
    with open("models/optimal_threshold.json", "w") as f:
        json.dump({
            "optimal_threshold": float(optimal["threshold"]),
            "total_cost_at_optimal": float(optimal["total_cost"]),
            "total_cost_at_naive_0.5": float(naive_row["total_cost"]),
            "expected_saving": float(savings),
            "loss_given_default_assumption": LOSS_GIVEN_DEFAULT,
            "net_interest_margin_assumption": NET_INTEREST_MARGIN,
            "test_pr_auc": float(pr_auc),
        }, f, indent=2)

    print("\nsaved threshold sweep to models/threshold_cost_sweep.csv")
