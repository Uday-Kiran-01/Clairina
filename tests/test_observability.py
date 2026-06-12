# tests/test_observability.py
import pandas as pd
from clairina.observability import check_schema, check_missingness, check_negative_spend

def test_observability_basic():
    df = pd.read_csv("data/prescient_synthetic_1000.csv", parse_dates=["date"])
    s = check_schema(df, ["date","revenue"])
    assert s["ok"]
    miss = check_missingness(df)
    assert "missing_rates" in miss
    neg = check_negative_spend(df)
    assert "negative_counts" in neg
