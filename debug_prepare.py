import traceback
print("Running prepare_features debug")
try:
    import scripts.prepare_features as pf
    pf.prepare()
    print("prepare_features ran OK")
except Exception:
    traceback.print_exc()
