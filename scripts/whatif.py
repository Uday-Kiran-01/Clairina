# scripts/whatif.py
import pandas as pd
import numpy as np
import joblib
from clairina.mmm import fit_mmm_ols, load_model
from pathlib import Path

FEAT_FILE = "data/prescient_features.csv"
MODEL_FILE = "models/mmm_ols.pkl"

def build_x_from_row(row, X_cols):
    x = []
    for name in X_cols:
        if name == "const":
            x.append(1.0)
        elif name in row.index:
            x.append(float(row[name]))
        elif name.startswith("log1p_"):
            ch = name.replace("log1p_","")
            x.append(float(row.get(name, np.log1p(row.get(ch,0.0)))))
        else:
            x.append(float(row.get(name, 0.0)))
    return x

def run_whatif(feat_file=FEAT_FILE, model_file=MODEL_FILE, channel="spend_google", pct=0.10):
    df = pd.read_csv(feat_file, parse_dates=["date"])
    if Path(model_file).exists():
        model = load_model(model_file)
    else:
        channels = [c for c in df.columns if c.startswith("spend_") and "_lag" not in c]
        model, _ = fit_mmm_ols(df, channels=channels, include_flags=True)

    X_cols = model.model.exog_names
    last_row = df.iloc[-1].copy()

    base_x = build_x_from_row(last_row, X_cols)
    pred_base = float(model.predict(base_x))

    scen = last_row.copy()
    scen[channel] = scen[channel] * (1.0 + pct)
    scen[f"log1p_{channel}"] = np.log1p(scen[channel])
    scen_x = build_x_from_row(scen, X_cols)
    pred_scen = float(model.predict(scen_x))

    print("Base predicted revenue (last row):", round(pred_base,2))
    print(f"Scenario (+{int(pct*100)}% {channel}):", round(pred_scen,2))
    print("Delta:", round(pred_scen - pred_base,2))

    # compute marginal revenue per $ at mean spend for quick ROAS
    channels = sorted([c for c in df.columns if c.startswith("spend_") and "_lag" not in c])
    means = {ch: float(df[ch].mean()) for ch in channels}
    params = model.params.to_dict()
    print("\nChannel    mean_spend    beta    marginal_rev_per_$    est_roas")
    for ch in channels:
        key = f"log1p_{ch}"
        beta = params.get(key, 0.0)
        spend = means[ch]
        marginal = beta / (1.0 + spend) if (1.0 + spend) != 0 else 0.0
        print(f"{ch:12} {spend:12.2f} {beta:10.3f} {marginal:20.4f} {marginal:12.4f}")

if __name__ == "__main__":
    run_whatif()
