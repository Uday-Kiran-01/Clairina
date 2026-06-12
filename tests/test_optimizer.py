# tests/test_optimizer.py
import pandas as pd
from clairina.mmm import prepare_features, fit_mmm_ols
from clairina.optimizer import optimize_budget_from_ols

def test_optimizer_feasible():
    df = pd.read_csv("data/prescient_synthetic_1000.csv", parse_dates=["date"])
    channels = [c for c in df.columns if c.startswith("spend_")]
    df2 = prepare_features(df, channels=channels)
    model, _ = fit_mmm_ols(df2, channels=channels)
    base = df2.iloc[-1].to_dict()
    total = float(base["total_spend"])
    res = optimize_budget_from_ols(model, base, channels, total_budget=total)
    assert res.success
    assert abs(res.x.sum() - total) < 1e-3
