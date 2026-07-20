# ML Workflow implemented

This repository now includes a runnable compact workflow that implements the common
ML engineering steps (except monitoring):

- EDA: basic shape and missing-value summary
- Data cleaning: median imputation and IQR outlier clipping
- Feature engineering: uses existing adstock/Hill transforms from `clairina/mmm.py`
- Feature selection: VIF-based removal of highly collinear features
- Train/Test split: time-based split (latest 20% as test)
- Model selection & training: `Ridge` and `RandomForestRegressor` via `GridSearchCV` with `TimeSeriesSplit`
- Hyperparameter tuning: grid search over a small set of params
- Evaluation: MAE, RMSE, R² saved to `data/model_eval.json`
- Model interpretation: coefficients/importances available from the saved model

Run the workflow:

```powershell
python scripts/ml_workflow.py
```

Artifacts produced:
- `models/model_sklearn.pkl` — best trained scikit-learn pipeline
- `data/model_eval.json` — EDA, VIF table, and evaluation metrics
