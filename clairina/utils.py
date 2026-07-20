"""Utility helpers for data cleaning, VIF, and time-series splits."""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def impute_missing_median(df, cols=None):
    df = df.copy()
    if cols is None:
        cols = df.columns
    for c in cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def clip_outliers_iqr(df, cols=None, factor=1.5):
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def compute_vif(df, features):
    """Compute VIF for a list of numeric feature columns.

    Returns DataFrame with columns: feature, vif
    """
    X = df[features].copy()
    # add small constant to avoid singularities
    X = X.fillna(0.0)
    X['__const__'] = 1.0
    vif_vals = []
    for i, col in enumerate(features):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = float('nan')
        vif_vals.append({'feature': col, 'vif': float(vif)})
    return pd.DataFrame(vif_vals).sort_values('vif', ascending=False)


def time_train_test_split(df, date_col='date', test_ratio=0.2):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test
