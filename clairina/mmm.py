# clairina/mmm.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import joblib

def adstock(series, rate=0.5):
    s = np.asarray(series, dtype=float)
    out = np.zeros_like(s)
    for t in range(len(s)):
        out[t] = s[t] + (out[t-1] * rate if t > 0 else 0.0)
    return out

def hill_saturation(x, alpha=1.0, gamma=1.0):
    x = np.asarray(x, dtype=float)
    num = np.power(x, alpha)
    den = num + np.power(gamma, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, 0.0, num / den)
    return out

def prepare_features(df, channels=None, use_adstock=True, adstock_rate=0.12,
                     use_saturation=False, hill_alpha=1.0, hill_gamma=1.0,
                     add_lags=True, lags=1):
    df = df.copy()
    if channels is None:
        channels = [c for c in df.columns if c.startswith("spend_")]
    for ch in channels:
        raw = df[ch].fillna(0.0).to_numpy(dtype=float)
        x = raw.copy()
        if use_adstock:
            x = adstock(raw, rate=adstock_rate)
        if use_saturation:
            sat = hill_saturation(x, alpha=hill_alpha, gamma=hill_gamma)
            scale = raw.max() if raw.max() > 0 else 1.0
            x = sat * scale
        df[f"adstock_{ch}"] = np.round(x, 6)
        df[f"log1p_{ch}"] = np.log1p(x)
        if add_lags:
            for L in range(1, lags+1):
                df[f"{ch}_lag{L}"] = df[ch].shift(L).fillna(0.0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["dayofweek"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
    return df

def prepare_X_y_for_ols(df, channels=None, include_flags=True):
    if channels is None:
        channels = [c for c in df.columns if c.startswith("spend_")]
    X_cols = [f"log1p_{ch}" for ch in channels]
    if include_flags:
        for f in ["promo_flag", "holiday_flag", "dayofweek", "month"]:
            if f in df.columns:
                X_cols.append(f)
    X = df[X_cols].fillna(0.0)
    X = sm.add_constant(X, has_constant='add')
    y = df["revenue"]
    return X, y, X_cols

def fit_mmm_ols(df, channels=None, include_flags=True):
    X, y, X_cols = prepare_X_y_for_ols(df, channels, include_flags)
    model = sm.OLS(y, X).fit()
    return model, X_cols

def summarize_model(model):
    """
    Return a safe summary dict with params, CI lower/upper, and rsq.
    Uses model.conf_int() (DataFrame) to extract CIs by row/param name.
    """
    params = model.params.to_dict()
    # conf_int is a DataFrame with columns [0, 1] (lower, upper)
    conf_int = model.conf_int()
    ci_lower = {}
    ci_upper = {}
    for param in params.keys():
        try:
            lower = conf_int.loc[param, 0]
            upper = conf_int.loc[param, 1]
        except Exception:
            # fallback if indexing differs
            row = conf_int.loc[param]
            lower = row.iloc[0]
            upper = row.iloc[1]
        ci_lower[param] = float(lower)
        ci_upper[param] = float(upper)

    return {
        "params": {k: float(v) for k, v in params.items()},
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rsq": float(model.rsquared)
    }

def save_model(model, path="models/mmm_ols.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path="models/mmm_ols.pkl"):
    return joblib.load(path)
