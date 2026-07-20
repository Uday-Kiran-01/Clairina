"""Run a compact ML workflow: EDA -> cleaning -> selection -> train/tune -> eval

Usage: python scripts/ml_workflow.py
"""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

from clairina.utils import impute_missing_median, clip_outliers_iqr, compute_vif, time_train_test_split
from clairina.mmm import prepare_features

ROOT = Path(".")
FEAT_CSV = ROOT / "data" / "prescient_features.csv"
MODEL_OUT = ROOT / "models" / "model_sklearn.pkl"
METRICS_OUT = ROOT / "data" / "model_eval.json"


def run():
    # Ensure features exist
    if not FEAT_CSV.exists():
        print("Prepared features not found — running scripts.prepare_features.prepare() to generate them.")
        prepare_features()

    df = pd.read_csv(FEAT_CSV, parse_dates=["date"]) if FEAT_CSV.exists() else None
    if df is None:
        raise FileNotFoundError("No feature file available at data/prescient_features.csv")

    # Basic EDA summary
    eda = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_per_column": df.isnull().sum().to_dict()
    }

    # Cleaning
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = impute_missing_median(df, cols=num_cols)
    df_clean = clip_outliers_iqr(df_clean, cols=num_cols)

    # Feature selection via VIF on numeric predictors (exclude target)
    target = "revenue"
    predictors = [c for c in df_clean.columns if c.startswith("log1p_") or c.startswith("adstock_")]
    # fallback to spend_ columns if no transformed columns
    if not predictors:
        predictors = [c for c in df_clean.columns if c.startswith("spend_")]

    vif_df = compute_vif(df_clean, predictors)
    # drop features with very high VIF (>10)
    high_vif = vif_df[vif_df.vif > 10]["feature"].tolist()
    selected = [f for f in predictors if f not in high_vif]
    if not selected:
        selected = predictors[:min(6, len(predictors))]

    # Train/test split (time-based)
    train_df, test_df = time_train_test_split(df_clean, date_col="date", test_ratio=0.2)

    X_train = train_df[selected]
    y_train = train_df[target]
    X_test = test_df[selected]
    y_test = test_df[target]

    # Models and hyperparams
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    param_grid = [
        {"model": [Ridge()], "model__alpha": [0.1, 1.0, 10.0]},
        {"model": [RandomForestRegressor(random_state=42, n_jobs=-1)], "model__n_estimators": [50, 100], "model__max_depth": [5, 10]}
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    preds = best.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "r2": float(r2_score(y_test, preds)),
        "best_params": gs.best_params_
    }

    # Save artifacts
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_OUT)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps({"eda": eda, "vif": vif_df.to_dict(orient="records"), "metrics": metrics}, indent=2))

    print("Workflow complete. Model saved to:", MODEL_OUT)
    print("Metrics:\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
