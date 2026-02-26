"""
classifier.py
-------------
Loads the trained models from disk and exposes a single predict() function
that the API calls for every incoming ticket.

Imports TextPreprocessor from the shared preprocessor module so that
pickle can deserialize the models correctly at load time.
"""

import os
import pickle

from preprocessor import TextPreprocessor  # noqa: F401 — required for pickle to deserialize models

# ------------------------------------------------------------------------------
# Load models at startup (only once, not on every request)
# ------------------------------------------------------------------------------

MODEL_DIR = "model"

with open(os.path.join(MODEL_DIR, "priority_model.pkl"), "rb") as f:
    priority_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "area_model.pkl"), "rb") as f:
    area_model = pickle.load(f)


# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

def predict(text: str) -> dict:
    """
    Classifies a support ticket and returns priority, area, and confidence.

    Parameters
    ----------
    text : Raw ticket description written by the user

    Returns
    -------
    dict with keys:
        ticket               : original input text
        priority             : predicted priority (Alta / Media / Baja)
        priority_confidence  : model confidence as a percentage string
        area                 : predicted area (Redes / Hardware / Accesos)
        area_confidence      : model confidence as a percentage string
    """
    priority = priority_model.predict([text])[0]
    area     = area_model.predict([text])[0]

    priority_proba = priority_model.predict_proba([text])[0]
    area_proba     = area_model.predict_proba([text])[0]

    return {
        "ticket":              text,
        "priority":            priority,
        "priority_confidence": f"{max(priority_proba) * 100:.1f}%",
        "area":                area,
        "area_confidence":     f"{max(area_proba) * 100:.1f}%",
    }