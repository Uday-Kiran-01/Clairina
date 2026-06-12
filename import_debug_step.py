import time, traceback
print("IMPORT DEBUG START")
try:
    print("importing numpy...")
    import numpy as np
    print(" -> numpy OK", np.__version__)
except Exception:
    traceback.print_exc()
    raise
try:
    print("importing pandas...")
    import pandas as pd
    print(" -> pandas OK", pd.__version__)
except Exception:
    traceback.print_exc()
    raise
try:
    print("importing statsmodels.api as sm...")
    import statsmodels.api as sm
    print(" -> statsmodels OK", sm.__version__)
except Exception:
    traceback.print_exc()
    raise
try:
    print("importing joblib...")
    import joblib
    print(" -> joblib OK", joblib.__version__)
except Exception:
    traceback.print_exc()
    raise
try:
    print("importing clairina.mmm module now...")
    import importlib
    m = importlib.import_module("clairina.mmm")
    print(" -> clairina.mmm imported:", m)
except Exception:
    traceback.print_exc()
    raise
print("IMPORT DEBUG DONE")
