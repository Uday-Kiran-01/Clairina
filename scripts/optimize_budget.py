# scripts/optimize_budget.py
"""
Optimize budget allocation using:
- Grid search (coarse, explainable)
- SciPy nonlinear constrained optimizer (fast, continuous)

Outputs best allocation, expected revenue, and lift vs baseline.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from math import log1p
from clairina.mmm import load_model

# SciPy optimizer (continuous)
from scipy.optimize import minimize

MODEL_PATH = Path("models/mmm_ols.pkl")
FEAT_PATH = Path("data/prescient_features.csv")

def predict_revenue_from_spends(beta_dict, const, spends):
    # spends: dict channel -> spend (float)
    # beta_dict: dict channel_key -> beta (for log1p_spend_<channel>)
    # const: intercept
    s = const
    for ch, val in spends.items():
        key = f"log1p_{ch}"
        beta = beta_dict.get(key, 0.0)
        s += beta * np.log1p(val)
    return float(s)

def load_model_info(model_path=MODEL_PATH):
    model = joblib.load(model_path)
    params = model.params.to_dict()
    intercept = params.get("const", 0.0)
    return model, params, intercept

def baseline_spends_from_data(feat_path=FEAT_PATH):
    df = pd.read_csv(feat_path, parse_dates=["date"])
    channels = sorted([c for c in df.columns if c.startswith("spend_") and "_lag" not in c])
    means = {ch: float(df[ch].mean()) for ch in channels}
    total = sum(means.values())
    return channels, means, total

# ----------------------
# Grid search optimizer
# ----------------------
def grid_optimize(channels, base_spends, total_budget, params, intercept,
                  steps_per_channel=8, min_share=0.0):
    """
    Grid search: distribute total_budget into channels with discrete steps.
    steps_per_channel controls coarseness (8 => each channel gets multiples of total_budget/(steps_per_channel*len(channels))).
    Return best allocation dict and predicted revenue.
    """
    n = len(channels)
    # discrete granularity
    gran = total_budget / (steps_per_channel * n)
    # create grid values for each channel (0..steps_per_channel*n) but cap to reasonable range
    values = [np.arange(0, steps_per_channel+1) * gran for _ in channels]  # coarse per-channel
    best = {"rev": -1e18, "alloc": None}
    # To keep search reasonable, sample combinations using random sampling if dimension too high
    max_combinations = (steps_per_channel + 1) ** n
    if max_combinations > 200000:
        # use randomized search of 100k samples
        import random
        samples = 100000
        for _ in range(samples):
            shares = np.random.dirichlet(np.ones(n))
            allocation = {channels[i]: float(shares[i] * total_budget) for i in range(n)}
            rev = predict_revenue_from_spends(params, intercept, allocation)
            if rev > best["rev"]:
                best = {"rev": rev, "alloc": allocation}
        return best
    else:
        # full grid (only feasible for small n/steps)
        from itertools import product
        for combo in product(*(vals for vals in values)):
            alloc = {channels[i]: float(combo[i]) for i in range(n)}
            # enforce sum to total_budget by scaling
            ssum = sum(alloc.values())
            if ssum == 0:
                continue
            for k in alloc:
                alloc[k] = alloc[k] * (total_budget / ssum)
            rev = predict_revenue_from_spends(params, intercept, alloc)
            if rev > best["rev"]:
                best = {"rev": rev, "alloc": alloc}
        return best

# ----------------------
# SciPy continuous optimizer
# ----------------------
def scipy_optimize(channels, base_spends, total_budget, params, intercept, bounds=None, initial=None):
    """
    Use SciPy to maximize predicted revenue with constraint sum(spends)=total_budget.
    We'll minimize negative revenue.
    """
    n = len(channels)
    if initial is None:
        initial = np.array([base_spends[ch] for ch in channels], dtype=float)
        # scale to budget
        s0 = initial.sum()
        if s0 == 0:
            initial = np.array([total_budget / n] * n)
        else:
            initial = initial * (total_budget / s0)

    if bounds is None:
        bounds = [(0.0, total_budget) for _ in channels]

    def obj(x):
        alloc = {channels[i]: float(max(0.0, x[i])) for i in range(n)}
        return -predict_revenue_from_spends(params, intercept, alloc)

    cons = ({'type': 'eq', 'fun': lambda x: float(np.sum(x) - total_budget)})
    res = minimize(obj, initial, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 1000})
    alloc = {channels[i]: float(max(0.0, res.x[i])) for i in range(n)}
    rev = predict_revenue_from_spends(params, intercept, alloc)
    return {"rev": rev, "alloc": alloc, "success": res.success, "message": res.message}

# ----------------------
# Runner for scenarios
# ----------------------
def run_scenarios(use_grid=True, use_scipy=True, budgets=None, steps_per_channel=6):
    model, params, intercept = load_model_info()
    channels, base_means, base_total = baseline_spends_from_data()
    # budgets: list of budgets to evaluate
    if budgets is None:
        budgets = {
            "current_avg": base_total,
            "fixed_5000": 5000.0,
            "low_med_high": [0.7 * base_total, base_total, 1.3 * base_total]
        }
    results = {}
    # normalize params keys: ensure dict contains log1p_spend_{channel} keys
    # params already contains full model params
    for label, bud in (budgets.items() if not isinstance(budgets, list) else budgets):
        pass
    # unify: create list of (name, value) tuples for budgets
    scenario_list = []
    if isinstance(budgets, dict):
        for k, v in budgets.items():
            if isinstance(v, list):
                for i, vv in enumerate(v):
                    scenario_list.append((f"{k}_{i}", float(vv)))
            else:
                scenario_list.append((k, float(v)))
    elif isinstance(budgets, (list, tuple)):
        for i, v in enumerate(budgets):
            scenario_list.append((f"scenario_{i}", float(v)))
    else:
        raise ValueError("Invalid budgets type")

    for name, bud in scenario_list:
        entry = {"budget": float(bud)}
        if use_grid:
            grid_res = grid_optimize(channels, base_means, bud, params, intercept, steps_per_channel=steps_per_channel)
            entry["grid"] = {"rev": grid_res["rev"], "alloc": grid_res["alloc"]}
        if use_scipy:
            sc_res = scipy_optimize(channels, base_means, bud, params, intercept)
            entry["scipy"] = {"rev": sc_res["rev"], "alloc": sc_res["alloc"], "success": sc_res.get("success")}
        # baseline revenue (current allocation scaled to bud)
        # baseline here we compute predicted rev at current mean spends scaled proportionally to bud
        cur_alloc_scaled = {ch: base_means[ch] * (bud / base_total) for ch in channels}
        entry["baseline"] = {"rev": predict_revenue_from_spends(params, intercept, base_means),
                             "scaled_baseline_rev": predict_revenue_from_spends(params, intercept, cur_alloc_scaled),
                             "current_total": base_total}
        results[name] = entry

    # Save results
    out = Path("data/optimization_results.json")
    out.write_text(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    # default budgets: current avg and 5000 and 3-levels
    model, params, intercept = load_model_info()
    channels, base_means, base_total = baseline_spends_from_data()
    print("Channels:", channels)
    print("Current total spend (avg):", round(base_total,2))
    budgets = {"current_avg": base_total, "fixed_5000": 5000.0, "multi": [0.7*base_total, base_total, 1.3*base_total]}
    res = run_scenarios(use_grid=True, use_scipy=True, budgets=budgets, steps_per_channel=6)
    print("Saved optimization_results.json — scenarios:", list(res.keys()))
