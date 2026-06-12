# scripts/prepare_features.py
import pandas as pd
import numpy as np
from clairina.mmm import adstock, hill_saturation
from pathlib import Path

IN = "data/prescient_synthetic_1000.csv"
OUT = "data/prescient_features.csv"

def prepare(df_path=IN, out_path=OUT, use_adstock=True, adstock_rate=0.12,
            use_saturation=False, hill_alpha=1.0, hill_gamma=1.0, add_lags=True, lags=1):
    df = pd.read_csv(df_path, parse_dates=["date"])
    channels = [c for c in df.columns if c.startswith("spend_")]
    for ch in channels:
        raw = df[ch].fillna(0.0).values
        x = raw.copy()
        if use_adstock:
            x = adstock(raw, rate=adstock_rate)
        if use_saturation:
            sat = hill_saturation(x, alpha=hill_alpha, gamma=hill_gamma)
            if raw.max() > 0:
                x = sat * raw.max()
            else:
                x = sat
        df[f"adstock_{ch}"] = np.round(x, 6)
        df[f"log1p_{ch}"] = np.log1p(x)
        if add_lags:
            for L in range(1, lags+1):
                df[f"{ch}_lag{L}"] = df[ch].shift(L).fillna(0.0)
    # date features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote features to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    prepare()
