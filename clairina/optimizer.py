# clairina/optimizer.py
import numpy as np
from scipy.optimize import minimize

def optimize_budget_from_ols(model, base_row, channels, total_budget, min_share=0.0, max_share=1.0):
    param_names = list(model.params.index)

    def predict_rev(spends):
        values = {}
        for i, ch in enumerate(channels):
            values[f"log1p_{ch}"] = np.log1p(spends[i])
        for name in param_names:
            if name not in values and name != "const":
                values[name] = float(base_row.get(name, base_row.get(name.replace("log1p_",""), 0.0)))
        ordered_x = []
        for name in param_names:
            if name == "const":
                ordered_x.append(1.0)
            else:
                ordered_x.append(values.get(name, 0.0))
        val = float(np.dot(model.params.values, np.array(ordered_x)))
        return -val

    x0 = np.array([float(base_row.get(ch, total_budget/len(channels))) for ch in channels])
    bounds = [(min_share * total_budget, max_share * total_budget) for _ in channels]
    cons = ({'type':'eq', 'fun': lambda x: x.sum() - total_budget})
    res = minimize(predict_rev, x0, bounds=bounds, constraints=cons)
    return res
