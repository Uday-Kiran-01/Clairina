# Running PrescientLens locally

Prerequisites
- Python 3.10+ recommended
- Git (optional)

Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

Run demo Streamlit app

```powershell
streamlit run streamlit_app.py
```

Run tests

```powershell
pytest -q
```

Notes
- If packages fail to install, try upgrading pip: `python -m pip install -U pip`.
- The demo expects sample data in the `data/` folder; see `examples/prescient_demo.py` for usage.
