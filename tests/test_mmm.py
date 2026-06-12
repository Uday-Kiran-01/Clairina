# tests/test_mmm.py
import pandas as pd
from clairina.mmm import adstock, hill_saturation, prepare_features, fit_mmm_ols

def test_adstock_simple():
    s = [1,0,0,0]
    out = adstock(s, rate=0.5)
    assert out[0] == 1
    assert out[1] >= 0 and out[1] <= 1

def test_hill_saturation_basic():
    x = [0,1,10]
    out = hill_saturation(x, alpha=1, gamma=1)
    assert out[0] == 0.0
    assert 0 <= out[1] <= 1

def test_prepare_and_fit(tmp_path):
    # create tiny df
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": dates,
        "spend_google": [10]*10,
        "spend_meta": [5]*10,
        "spend_tiktok":[1]*10,
        "spend_email":[2]*10,
        "spend_tv":[3]*10,
        "spend_organic":[1]*10,
        "promo_flag":[0]*10,
        "holiday_flag":[0]*10,
        "revenue":[100 + i for i in range(10)]
    })
    df2 = prepare_features(df)
    model, cols = fit_mmm_ols(df2)
    assert hasattr(model, "params")
