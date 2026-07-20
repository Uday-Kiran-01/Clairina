import pandas as pd
import numpy as np

from clairina.utils import impute_missing_median, clip_outliers_iqr, compute_vif, time_train_test_split


def test_impute_and_clip():
    df = pd.DataFrame({
        'a': [1.0, np.nan, 3.0, 1000.0],
        'b': [10.0, 20.0, np.nan, 40.0],
        'date': pd.date_range('2024-01-01', periods=4)
    })
    df2 = impute_missing_median(df, cols=['a', 'b'])
    assert not df2['a'].isnull().any()
    assert not df2['b'].isnull().any()

    df3 = clip_outliers_iqr(df2, cols=['a', 'b'])
    # clipping should reduce extremely large value
    assert df3['a'].max() < 1000.0


def test_vif_and_time_split():
    # Construct correlated features
    n = 50
    x1 = np.linspace(1, 100, n)
    x2 = x1 * 0.9 + np.random.normal(0, 1, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'date': pd.date_range('2024-01-01', periods=n)})
    vif = compute_vif(df, ['x1', 'x2'])
    assert 'feature' in vif.columns and 'vif' in vif.columns

    tr, te = time_train_test_split(df, date_col='date', test_ratio=0.2)
    assert len(tr) + len(te) == n
