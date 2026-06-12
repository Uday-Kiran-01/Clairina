# clairina/bayesian_mmm.py
import pymc as pm
import arviz as az
import pandas as pd
import numpy as np

def bayesian_mmm_fit(df, channels=None, draws=1000, tune=1000, target_accept=0.9):
    if channels is None:
        channels = [c for c in df.columns if c.startswith("spend_") and "_lag" not in c]
    X = np.column_stack([np.log1p(df[ch].values) for ch in channels])
    y = df["revenue"].values
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=1e3)
        betas = pm.Normal("betas", mu=0, sigma=500, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1e3)
        mu = intercept + pm.math.dot(X, betas)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, cores=2)
    summary = az.summary(trace)
    return model, trace, summary
