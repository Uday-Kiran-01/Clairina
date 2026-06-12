print("DEBUG: starting")
import traceback
try:
    print("DEBUG: importing scripts.prepare_features")
    import scripts.prepare_features as pf
    print("DEBUG: calling prepare()")
    pf.prepare()
    print("DEBUG: DONE — prepare_features ran OK")
except Exception:
    print("DEBUG: error occurred")
    traceback.print_exc()
