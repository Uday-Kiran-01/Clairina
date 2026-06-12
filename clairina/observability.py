# clairina/observability.py
import logging, os
LOGGER = logging.getLogger("clairina")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    h = logging.FileHandler("logs/pipeline.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(h)

def log_step(step, status="OK", msg=""):
    LOGGER.info("%s | %s | %s", step, status, msg)
