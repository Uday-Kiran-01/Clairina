# clairina/data_utils.py

import pandas as pd
import numpy as np

def generate_synthetic_data(start_date='2024-01-01', end_date='2024-06-30', seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq='D')
    n = len(dates)
    
    # Simulate daily ad spend per channel
    spend_google = np.random.randint(500, 2000, n)
    spend_meta = np.random.randint(300, 1500, n)
    spend_tiktok = np.random.randint(200, 1200, n)
    spend_tv = np.random.randint(1000, 5000, n)
    spend_email = np.random.randint(100, 800, n)
    
    # Promotions / holidays
    promotion_flag = np.random.choice([0,1], n, p=[0.85, 0.15])
    holiday_flag = np.random.choice([0,1], n, p=[0.9, 0.1])
    
    # Base revenue + effects
    revenue = (
        50 
        + 0.05 * np.log(spend_google + 1) * 1000
        + 0.04 * np.log(spend_meta + 1) * 1000
        + 0.03 * np.log(spend_tiktok + 1) * 1000
        + 0.02 * np.log(spend_tv + 1) * 1000
        + 0.01 * np.log(spend_email + 1) * 1000
        + 500 * promotion_flag
        + 300 * holiday_flag
        + np.random.normal(0, 200, n)  # noise
    )
    
    data = pd.DataFrame({
        'date': dates,
        'spend_google': spend_google,
        'spend_meta': spend_meta,
        'spend_tiktok': spend_tiktok,
        'spend_tv': spend_tv,
        'spend_email': spend_email,
        'promotion_flag': promotion_flag,
        'holiday_flag': holiday_flag,
        'revenue': revenue
    })
    
    return data
