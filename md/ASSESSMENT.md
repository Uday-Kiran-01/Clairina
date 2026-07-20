# Assessment: ML approach

Summary:

This project uses a classical econometric / regression-based marketing-mix
model (MMM) as its primary approach. Key indicators:

- `clairina/mmm.py` implements adstock, Hill saturation, `log1p` transforms,
  and fits an OLS model using `statsmodels` (`fit_mmm_ols`).
- `clairina/optimizer.py` derives budget allocations by optimizing the OLS
  model's linear predictions via `scipy.optimize.minimize`.
- There is an optional Bayesian variant in `clairina/bayesian_mmm.py` using
  `pymc` (MCMC sampling), but the default pipeline (and Streamlit UI) uses OLS.

Verdict: this is a standard ("crisp") statistical ML approach — feature
engineering + linear regression with interpretable parameters and CI. It's not
using black-box ML (tree ensembles or neural networks) for core modeling.

If you want, I can:

- Add a short section to `md/README.md` summarizing this assessment.
- Propose steps to add a tree-based or ML pipeline (sklearn) if you want modern models.
