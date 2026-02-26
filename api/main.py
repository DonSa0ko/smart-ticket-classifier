"""
main.py
-------
FastAPI REST API for the Smart Ticket Classifier.

Endpoints:
    GET  /              Health check
    POST /classify      Classifies a support ticket

Usage:
    uvicorn api.main:app --reload

Then open: http://127.0.0.1:8000/docs
"""

import sys
import os

# Allow imports from the project root (classifier.py, model/)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classifier import predict

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Smart Ticket Classifier",
    description=(
        "NLP-powered API that classifies IT support tickets by priority "
        "(Alta / Media / Baja) and responsible area (Redes / Hardware / Accesos)."
    ),
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------

class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Description of the support ticket to classify",
        example="The WiFi in the warehouse is completely down"
    )

class TicketResponse(BaseModel):
    ticket:               str
    priority:             str
    priority_confidence:  str
    area:                 str
    area_confidence:      str

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Returns API status. Use this to verify the service is running."""
    return {"status": "ok", "service": "Smart Ticket Classifier"}


@app.post("/classify", response_model=TicketResponse, tags=["Classification"])
def classify_ticket(request: TicketRequest):
    """
    Classifies a support ticket and returns:
    - **priority**: Alta, Media, or Baja
    - **priority_confidence**: model confidence for the priority prediction
    - **area**: Redes, Hardware, or Accesos
    - **area_confidence**: model confidence for the area prediction
    """
    try:
        result = predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
