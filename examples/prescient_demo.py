# examples/prescient_demo.py
import pandas as pd
import numpy as np
from clairina.mmm import prepare_features, fit_mmm_ols, summarize_model
from clairina.insights import generate_insights

df = pd.read_csv("data/prescient_synthetic_1000.csv", parse_dates=["date"])
channels = [c for c in df.columns if c.startswith("spend_")]

df_feat = prepare_features(df, channels=channels, use_adstock=True, adstock_rate=0.12, add_lags=True)
model, cols = fit_mmm_ols(df_feat, channels=channels)
summary = summarize_model(model)
print("R-squared:", summary["rsq"])
print("Top params (log1p features):")
for ch in channels:
    print(ch, "->", summary["params"].get(f"log1p_{ch}", None))

baseline = {"total_spend": float(df_feat.total_spend.mean()), "revenue": float(df_feat.revenue.mean())}
ins = generate_insights(summary, channels, baseline, provider="mock")
print("\nLLM-style insight (mock):")
print(ins)
