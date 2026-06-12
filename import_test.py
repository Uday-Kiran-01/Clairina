import traceback
print("RUNNING import_test.py")
try:
    import importlib
    m = importlib.import_module("clairina.mmm")
    print("IMPORT OK, module:", m)
except Exception:
    print("IMPORT FAILED")
    traceback.print_exc()
