# scripts/run_mmm.py
import pandas as pd
from clairina.mmm import prepare_features, fit_mmm_ols, summarize_model, save_model
from pathlib import Path
import json

def detect_base_spend_channels(df):
    # detect spend_ columns but exclude any that are lag or derived (contain '_lag' or other suffixes)
    raw_spend = [c for c in df.columns if c.startswith("spend_")]
    base = []
    for c in raw_spend:
        # exclude lag columns like spend_google_lag1, spend_google_lag2 etc.
        if "_lag" in c:
            continue
        # exclude adstock/log1p columns if they accidentally start with spend_
        if c.count("_") > 1 and c.endswith(tuple(str(i) for i in range(1,10))):
            continue
        base.append(c)
    # ensure deterministic order
    base = sorted(list(set(base)))
    return base

def run_mmm():
    # Load prepared features (should already exist)
    features_path = Path("data/prescient_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Run prepare_features.py first — features file not found.")

    print("Loading features...")
    df = pd.read_csv(features_path)

    # Identify only base channels (exclude spend_*_lag* etc)
    channels = detect_base_spend_channels(df)
    print("Base channels detected:", channels)

    # If log1p_* columns are missing for any channel, run feature prep to create them
    missing_log1p = [f"log1p_{ch}" for ch in channels if f"log1p_{ch}" not in df.columns]
    if missing_log1p:
        print("Missing log1p features for channels:", missing_log1p)
        print("Re-running feature preparation to ensure features exist...")
        df = prepare_features(df, channels=channels, use_adstock=True, adstock_rate=0.12, add_lags=True)
        # reload df after feature creation (prepare_features returns df already)
    else:
        print("All required log1p features present.")

    # Fit OLS MMM
    print("Fitting OLS MMM model...")
    model, used_cols = fit_mmm_ols(df, channels=channels, include_flags=True)
    summary = summarize_model(model)

    # Save model
    save_model(model, "models/mmm_ols.pkl")

    # Save coefficients
    coefs_path = Path("data/mmm_coefs.json")
    with open(coefs_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved OLS model to models/mmm_ols.pkl")
    print(f"Saved coefficients to {coefs_path}")
    print(f"R-squared: {summary['rsq']:.4f}")

    print("\nTop parameters (log1p features):")
    params = summary["params"]
    # show top 6 channel impacts
    log_params = {k: v for k, v in params.items() if k.startswith("log1p_spend_")}
    for k, v in sorted(log_params.items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    run_mmm()
