# scripts/compute_marginal.py
import pandas as pd
import json
from clairina.mmm import load_model
from pathlib import Path

FEAT = Path("data/prescient_features.csv")
MODEL = Path("models/mmm_ols.pkl")

def main():
    if not FEAT.exists():
        raise FileNotFoundError("Run prepare_features.py first.")
    if not MODEL.exists():
        raise FileNotFoundError("Run run_mmm.py first.")
    df = pd.read_csv(FEAT, parse_dates=["date"])
    model = load_model(MODEL)

    # detect base channels
    channels = sorted([c for c in df.columns if c.startswith("spend_") and "_lag" not in c])
    means = {ch: float(df[ch].mean()) for ch in channels}

    params = model.params.to_dict()
    marginals = {}
    for ch in channels:
        key = f"log1p_{ch}"
        beta = params.get(key, 0.0)
        spend = means[ch]
        marginal_rev_per_dollar = beta / (1.0 + spend)  # derivative of beta * log1p(spend)
        marginals[ch] = {
            "beta": float(beta),
            "mean_spend": float(spend),
            "marginal_rev_per_$": float(marginal_rev_per_dollar),
            "estimated_roas": float(marginal_rev_per_dollar)  # revenue per additional $1
        }

    print("Channel marginal revenue (per $1 additional spend) at mean spend:")
    print("{:20} {:>12} {:>12} {:>15}".format("channel","mean_spend","beta","marginal_rev/$"))
    for ch, v in marginals.items():
        print(f"{ch:20} {v['mean_spend']:12.2f} {v['beta']:12.3f} {v['marginal_rev_per_$']:15.4f}")

    # save quick JSON
    out = Path("data/marginal_rev.json")
    out.write_text(json.dumps(marginals, indent=2))
    print(f"\nSaved marginal estimates to {out}")

if __name__ == "__main__":
    main()
