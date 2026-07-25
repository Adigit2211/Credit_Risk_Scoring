"""
Stage 2: modeling.

The imbalance-handling decision here is the one thing I want to flag
up front, because it's not what I originally expected going in, and the
"why" is worth more in an interview than the "what".

The textbook advice for an ~8% positive rate is "reweight or resample."
I tried both - class_weight='balanced' and SMOTE - and empirically
*neither helped PR-AUC*, and scale_pos_weight in LightGBM actively hurt
it (best_iteration collapsed from ~300 rounds to single digits once
reweighting was cranked up). See compare_imbalance_strategies() for the
numbers.

The reason, once you think about it, isn't a bug - PR-AUC (and ROC-AUC)
are *ranking* metrics: they evaluate how well the model orders applicants
by risk across every possible threshold, not how many it flags at any
one threshold. Reweighting the loss changes where the model's decision
boundary sits for a *fixed* unweighted threshold, but the model already
sees plenty of both classes over training to learn to rank them - forcing
extra gradient weight onto the rare class just distorts the probability
calibration without improving the ordering.

So the imbalance is real, but the *right place to handle it* isn't the
loss function - it's the threshold you pick after training, using the
actual business costs of a false negative vs a false positive. That's
what evaluate.py does. Models here are trained unweighted; imbalance is
handled downstream, at decision time, where it actually has business
meaning attached to it. I'm keeping the SMOTE/class-weight comparison
code in because "I tried the standard approach, measured it, and it
didn't hold up here" is a much better interview answer than either
blindly applying it or never having checked.

Second design choice: LightGBM over XGBoost, mainly for handling the
categorical columns natively (no one-hot explosion) and training
noticeably faster on a dataset this size - and native categoricals matter
because the same encoding needs to run at inference time in the API
without re-fitting anything.

Logistic regression is trained too, purely as an interpretability anchor -
"does the sign of each coefficient match domain intuition" is a sanity
check worth having even though it's not the deployed model.
"""

from __future__ import annotations

import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from src.data_pipeline import run_pipeline, get_feature_lists, prep_categoricals_for_lgbm, TARGET_COL

RANDOM_STATE = 42


def load_and_split(raw_path: str):
    df, imputation_stats = run_pipeline(raw_path)
    flists = get_feature_lists(df)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # stratified split matters here specifically because the target is
    # imbalanced (~8% positive) - a plain random split can, by chance,
    # shift that ratio enough between train/test to distort your PR-AUC
    # comparison. stratify pins the ratio in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, flists, imputation_stats


def build_lr_pipeline(flists, class_weight: str | None = None) -> Pipeline:
    """Logistic regression needs scaled numerics and one-hot categoricals -
    trees don't care about scale but LR very much does, which is itself
    a good thing to be able to explain (distance/gradient-based models
    need comparable feature ranges, split-based models don't).

    class_weight defaults to None (unweighted) - see the module docstring
    for why: reweighting didn't improve PR-AUC here, so imbalance is
    handled at threshold-selection time instead, not baked into training.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), flists.numeric + flists.binary_flag),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), flists.categorical),
    ])
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def compare_imbalance_strategies(X_train, y_train, flists, n_folds: int = 3) -> dict:
    """Runs a small CV comparison of three approaches on the baseline LR
    model: no reweighting, class_weight='balanced', and SMOTE. This is what
    the module docstring's claim is actually based on - not an assumption.
    Keeping it to LR + a few folds since the point is to demonstrate the
    trade-off cheaply, not to exhaustively grid search it.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    X_arr = X_train.reset_index(drop=True)
    y_arr = y_train.reset_index(drop=True)

    none_scores, weighted_scores, smote_scores = [], [], []
    none_seconds, weighted_seconds, smote_seconds = [], [], []

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
        y_tr, y_val = y_arr.iloc[train_idx], y_arr.iloc[val_idx]

        # arm 1: no reweighting at all
        t0 = time.time()
        pipe = build_lr_pipeline(flists, class_weight=None)
        pipe.fit(X_tr, y_tr)
        none_seconds.append(time.time() - t0)
        preds = pipe.predict_proba(X_val)[:, 1]
        none_scores.append(average_precision_score(y_val, preds))

        # arm 2: class-weighted
        t0 = time.time()
        pipe = build_lr_pipeline(flists, class_weight="balanced")
        pipe.fit(X_tr, y_tr)
        weighted_seconds.append(time.time() - t0)
        preds = pipe.predict_proba(X_val)[:, 1]
        weighted_scores.append(average_precision_score(y_val, preds))

        # arm 3: SMOTE - crucially, fit only on the training fold, never on
        # validation, and never before the split. Fitting it before
        # splitting is a classic leakage bug: synthetic minority points
        # generated from information that includes what ends up in your
        # "held out" set.
        t0 = time.time()
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), flists.numeric + flists.binary_flag),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), flists.categorical),
        ])
        X_tr_enc = preprocessor.fit_transform(X_tr)
        X_val_enc = preprocessor.transform(X_val)
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr_enc, y_tr)
        lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        lr.fit(X_tr_res, y_tr_res)
        smote_seconds.append(time.time() - t0)
        preds = lr.predict_proba(X_val_enc)[:, 1]
        smote_scores.append(average_precision_score(y_val, preds))

    return {
        "no_reweighting_pr_auc_mean": float(np.mean(none_scores)),
        "no_reweighting_pr_auc_std": float(np.std(none_scores)),
        "class_weight_pr_auc_mean": float(np.mean(weighted_scores)),
        "class_weight_pr_auc_std": float(np.std(weighted_scores)),
        "smote_pr_auc_mean": float(np.mean(smote_scores)),
        "smote_pr_auc_std": float(np.std(smote_scores)),
        "no_reweighting_avg_seconds": float(np.mean(none_seconds)),
        "class_weight_avg_seconds": float(np.mean(weighted_seconds)),
        "smote_avg_seconds": float(np.mean(smote_seconds)),
    }


def train_lightgbm(X_train, y_train, X_test, y_test, flists):
    # early stopping needs its own eval set, and it must NOT be the final
    # test set - using the test set here would mean "how many trees to
    # grow" gets chosen based on the exact data we later report metrics
    # on, which quietly inflates the reported score. So we carve a proper
    # validation split out of the training data instead and keep X_test
    # completely untouched until evaluate_model() runs on it at the end.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )

    X_tr_c, categories = prep_categoricals_for_lgbm(X_tr, flists)
    X_val_c, _ = prep_categoricals_for_lgbm(X_val, flists, categories=categories)

    # deliberately NOT setting scale_pos_weight here - tested it (see module
    # docstring), it collapsed best_iteration from ~300 rounds down to
    # single digits and PR-AUC dropped from ~0.24 to ~0.19. Leaving the
    # loss unweighted lets the model spend its full capacity on ranking
    # quality; the business's actual asymmetric cost preference gets
    # applied later, at threshold selection, where it belongs.
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=50,      # guards against leaves fit to a handful of noisy rows
        subsample=0.8,             # row subsampling per tree - cheap regularisation
        colsample_bytree=0.8,      # same idea per column
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_tr_c, y_tr,
        eval_set=[(X_val_c, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        categorical_feature=flists.categorical,
    )
    print(f"  best iteration: {model.best_iteration_}")
    return model, categories


def evaluate_model(model, X, y, is_lgbm: bool) -> dict:
    preds = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, preds)),
    }


if __name__ == "__main__":
    print("loading and splitting data...")
    X_train, X_test, y_train, y_test, flists, imputation_stats = load_and_split("data/raw/application_train.csv")
    print(f"train: {X_train.shape}, test: {X_test.shape}, positive rate: {y_train.mean():.4f}")

    print("\n--- SMOTE vs class-weighting comparison (3-fold CV, LR) ---")
    comparison = compare_imbalance_strategies(X_train, y_train, flists)
    for k, v in comparison.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- training baseline logistic regression ---")
    lr_pipeline = build_lr_pipeline(flists)
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, is_lgbm=False)
    print(f"  logistic regression test PR-AUC: {lr_metrics['pr_auc']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")

    print("\n--- training LightGBM ---")
    lgbm_model, categories = train_lightgbm(X_train, y_train, X_test, y_test, flists)
    X_test_c, _ = prep_categoricals_for_lgbm(X_test, flists, categories=categories)
    lgbm_metrics = evaluate_model(lgbm_model, X_test_c, y_test, is_lgbm=True)
    print(f"  LightGBM test PR-AUC: {lgbm_metrics['pr_auc']:.4f}, ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")

    print("\nsaving models...")
    joblib.dump(lr_pipeline, "models/logreg_baseline.pkl")
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    joblib.dump(flists, "models/feature_lists.pkl")
    with open("models/feature_columns.json", "w") as f:
        # exact column set + order the model was trained on - the API
        # reindexes every incoming request against this before prediction,
        # so any field the live form doesn't collect (e.g. FLAG_DOCUMENT_*)
        # shows up as a real column filled with NaN rather than a missing
        # column that would break LightGBM's predict() call outright
        json.dump(list(X_train.columns), f, indent=2)
    with open("models/lgbm_categories.json", "w") as f:
        json.dump(categories, f)
    with open("models/imputation_stats.json", "w") as f:
        json.dump(imputation_stats, f, indent=2)
    with open("models/metrics.json", "w") as f:
        json.dump({
            "logistic_regression": lr_metrics,
            "lightgbm": lgbm_metrics,
            "imbalance_comparison": comparison,
        }, f, indent=2)

    # also stash the test set (post-cleaning, pre-categorical-cast) so
    # evaluate.py can do the cost-threshold analysis without re-running
    # the whole pipeline
    X_test.assign(**{TARGET_COL: y_test}).to_parquet("data/processed/test_set.parquet", index=False)

    print("\ndone. models saved to models/")
