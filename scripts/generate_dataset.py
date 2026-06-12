# scripts/generate_dataset.py
import numpy as np, pandas as pd
from pathlib import Path

def generate_dataset(days=1000, seed=123, out_path="data/prescient_synthetic_1000.csv"):
    np.random.seed(seed)
    start_date = "2023-01-01"
    dates = pd.date_range(start_date, periods=days, freq="D")
    def gen_spend(base, scale):
        return np.random.gamma(base, scale, size=days) + np.linspace(0, scale*2, days)
    spend_google = gen_spend(5,200)
    spend_meta = gen_spend(4,150)
    spend_tiktok = gen_spend(2,80)
    spend_email = gen_spend(1.5,40)
    spend_tv = gen_spend(3,300)
    spend_organic = np.maximum(50 + 5*np.sin(np.linspace(0, 12.56, days)) + np.random.normal(0,10,days), 10)
    promo_flag = (np.random.rand(days) < 0.06).astype(int)
    holiday_dates = pd.to_datetime(["2023-01-01","2023-12-25","2023-11-23","2024-01-01","2024-12-25","2024-11-28"])
    holiday_flag = pd.Series(dates.isin(holiday_dates)).astype(int).values
    base = 500 + np.linspace(0,800,days)
    season = 1 + 0.12 * np.sin(2 * np.pi * dates.dayofyear / 365)
    coeff = {"google":180, "meta":150, "tiktok":90, "email":110, "tv":120, "organic":80}
    lag = 0.12
    rev = base * season.copy()
    rev += coeff["google"] * np.log1p(spend_google + lag * np.roll(spend_google,1))
    rev += coeff["meta"] * np.log1p(spend_meta + lag * np.roll(spend_meta,1))
    rev += coeff["tiktok"] * np.log1p(spend_tiktok + lag * np.roll(spend_tiktok,1))
    rev += coeff["email"] * np.log1p(spend_email + lag * np.roll(spend_email,1))
    rev += coeff["tv"] * np.log1p(spend_tv + lag * np.roll(spend_tv,1))
    rev += coeff["organic"] * np.log1p(spend_organic + lag * np.roll(spend_organic,1))
    rev += promo_flag * 1000 + holiday_flag * 800
    rev += np.random.normal(0,600,size=days)
    rev = np.maximum(rev,0).round(2)
    df = pd.DataFrame({
        "date": dates,
        "spend_google": np.round(spend_google,2),
        "spend_meta": np.round(spend_meta,2),
        "spend_tiktok": np.round(spend_tiktok,2),
        "spend_email": np.round(spend_email,2),
        "spend_tv": np.round(spend_tv,2),
        "spend_organic": np.round(spend_organic,2),
        "promo_flag": promo_flag,
        "holiday_flag": holiday_flag,
        "revenue": rev
    })
    df["total_spend"] = df[[c for c in df.columns if c.startswith("spend_")]].sum(axis=1).round(2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    generate_dataset()
