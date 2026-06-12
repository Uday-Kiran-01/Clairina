# streamlit_app.py (defensive, safe, won't produce blank page)
import os
import json
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PrescientLens — Safe App", layout="wide")
st.title("PrescientLens — Safe App (diagnostic)")

ROOT = Path(".")
DATA_RAW_PATH = ROOT / "data" / "prescient_synthetic_1000.csv"
FEAT_PATH = ROOT / "data" / "prescient_features.csv"
MODEL_PATH = ROOT / "models" / "mmm_ols.pkl"
MARG_PATH = ROOT / "data" / "marginals.json"
OPT_PATH = ROOT / "data" / "optimization_results.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL")

def safe_read_csv(p: Path):
    try:
        if not p.exists():
            return None, f"File not found: {p}"
        df = pd.read_csv(p, parse_dates=["date"], low_memory=False)
        return df, None
    except Exception as e:
        return None, f"Error reading {p}: {e}"

# Top-level status panel (always visible)
with st.sidebar:
    st.header("Status / Quick checks")
    st.write("- App loaded OK (no blocking import).")
    st.write(f"- OLLAMA_URL set: {'Yes' if OLLAMA_URL else 'No'}")
    st.write(f"- raw data exists: {'Yes' if DATA_RAW_PATH.exists() else 'No'}")
    st.write(f"- prepared features exists: {'Yes' if FEAT_PATH.exists() else 'No'}")
    st.write(f"- model file exists: {'Yes' if MODEL_PATH.exists() else 'No'}")
    st.write(f"- marginals exists: {'Yes' if MARG_PATH.exists() else 'No'}")
    st.write(f"- optimization exists: {'Yes' if OPT_PATH.exists() else 'No'}")
    st.markdown("---")
    st.write("If the page is blank, check the terminal where you ran `streamlit run` for tracebacks.")

# Show raw data if available
st.subheader("Raw dataset (safe preview)")
df_raw, err = safe_read_csv(DATA_RAW_PATH)
if err:
    st.warning(err)
    st.info("You can generate the synthetic dataset using `python scripts/generate_dataset.py`.")
else:
    st.dataframe(df_raw.head(6))

# Feature prep area (safe)
st.subheader("Prepare features (safe)")
try:
    if st.button("Run prepare_features (scripts.prepare_features.prepare)"):
        # lazy import to avoid failures at import time
        import importlib
        mod = importlib.import_module("scripts.prepare_features")
        mod.prepare()
        st.success("prepare_features completed. Check data/prescient_features.csv")
    else:
        st.write("Click the button to run feature preparation. This will create data/prescient_features.csv")
except Exception as e:
    st.error("prepare_features raised an exception:")
    st.code(traceback.format_exc())

# Load features (non-blocking)
st.subheader("Prepared features (if present)")
df_feat, err = safe_read_csv(FEAT_PATH)
if err:
    st.info("Prepared features not found. Run feature preparation above or run `python scripts/prepare_dataset.py`.")
else:
    st.dataframe(df_feat.head(6))

# Fit MMM (safe)
st.subheader("Train MMM (safe)")
try:
    if st.button("Fit OLS MMM (clairina.mmm.fit_mmm_ols)"):
        import importlib
        mm = importlib.import_module("clairina.mmm")
        if df_feat is None:
            st.error("No features available to fit. Run prepare_features first.")
        else:
            model, used_cols = mm.fit_mmm_ols(df_feat)
            summary = mm.summarize_model(model)
            # save artifacts
            Path("models").mkdir(exist_ok=True)
            import joblib
            joblib.dump(model, MODEL_PATH)
            with open(ROOT / "data" / "mmm_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            st.success(f"Model trained. R² = {summary.get('rsq'):.4f}")
            st.json(summary)
    else:
        st.write("Click to fit the OLS MMM model.")
except Exception:
    st.error("Error during model fit:")
    st.code(traceback.format_exc())

# Compute marginals (safe)
st.subheader("Compute marginals (safe)")
try:
    if st.button("Compute marginals (scripts.compute_marginal.compute_marginals)"):
        import importlib
        cm = importlib.import_module("scripts.compute_marginal")
        if df_feat is None:
            st.error("Need prepared features to compute marginals.")
        else:
            marg = cm.compute_marginals(df_feat, json.load(open(ROOT / "data" / "mmm_summary.json")))
            with open(MARG_PATH, "w") as f:
                json.dump(marg, f, indent=2)
            st.success("Marginals computed and saved.")
            st.json(marg)
    else:
        st.write("Click to compute marginals.")
except Exception:
    st.error("Error computing marginals:")
    st.code(traceback.format_exc())

# Run optimizer (safe)
st.subheader("Run optimizer (safe)")
try:
    if st.button("Run optimizer (scripts.optimize_budget.run_scenarios)"):
        import importlib
        ob = importlib.import_module("scripts.optimize_budget")
        res = ob.run_scenarios(use_grid=True, use_scipy=True, budgets={"current_avg": None}, steps_per_channel=6)
        Path("data").mkdir(exist_ok=True)
        with open(OPT_PATH, "w") as f:
            json.dump(res, f, indent=2)
        st.success("Optimization completed and saved to data/optimization_results.json")
        st.write(list(res.keys()))
    else:
        st.write("Click to run optimizer (may take a few seconds).")
except Exception:
    st.error("Error running optimizer:")
    st.code(traceback.format_exc())

# LLM explanation (optional, non-blocking)
st.subheader("LLM Explain (Ollama) - optional")
if st.button("Explain using Ollama (if configured)"):
    if not OLLAMA_URL:
        st.error("OLLAMA_URL not configured in environment. Example: set OLLAMA_URL=http://localhost:11434")
    else:
        try:
            # safe read of summary,marg,opt
            summ = None
            try:
                summ = json.load(open(ROOT / "data" / "mmm_summary.json"))
            except Exception:
                summ = {"note": "no summary file found"}
            marg = None
            try:
                marg = json.load(open(MARG_PATH))
            except Exception:
                marg = {"note": "no marginals file"}
            opt = None
            try:
                opt = json.load(open(OPT_PATH))
            except Exception:
                opt = {"note": "no optimization file"}

            prompt = f"Summarize in 3 bullets. Summary: {summ}. Marginals: {marg}. Optimization: {list(opt.keys())}"
            resp = None
            try:
                import requests
                r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model":"mistral", "prompt":prompt}, timeout=20)
                r.raise_for_status()
                resp = r.json().get("output", "(no output)")
            except Exception as e:
                resp = f"Ollama request failed: {e}"
            st.write(resp)
        except Exception:
            st.error("LLM step failed:")
            st.code(traceback.format_exc())

st.caption("This safe app will always render. Use the buttons to run pipeline steps. Check the terminal where you started Streamlit for detailed logs if something fails.")
