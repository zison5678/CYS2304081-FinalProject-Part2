import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"


class IncidentInput(BaseModel):
    country: str = Field(min_length=1)
    year: int = Field(ge=2015, le=2030)
    attack_type: str = Field(min_length=1)
    target_industry: str = Field(min_length=1)
    financial_loss_million: float = Field(ge=0)
    affected_users: int = Field(ge=0)
    attack_source: str = Field(min_length=1)
    vulnerability_type: str = Field(min_length=1)
    defense_mechanism: str = Field(min_length=1)


class PredictionOutput(BaseModel):
    predicted_resolution_hours: float
    predicted_resolution_days: float
    model: str
    unit: str


def create_input_row(data: IncidentInput) -> dict[str, Any]:
    years_since_2015 = data.year - 2015
    log_financial_loss = math.log1p(data.financial_loss_million)
    log_affected_users = math.log1p(data.affected_users)

    if data.affected_users > 0:
        loss_per_user_usd = (
            data.financial_loss_million * 1_000_000
        ) / data.affected_users
    else:
        loss_per_user_usd = 0.0

    return {
        "country": data.country.strip(),
        "attack_type": data.attack_type.strip(),
        "target_industry": data.target_industry.strip(),
        "attack_source": data.attack_source.strip(),
        "vulnerability_type": data.vulnerability_type.strip(),
        "defense_mechanism": data.defense_mechanism.strip(),
        "years_since_2015": int(years_since_2015),
        "log_financial_loss": float(log_financial_loss),
        "log_affected_users": float(log_affected_users),
        "loss_per_user_usd": float(loss_per_user_usd),
    }


if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model_artifact = joblib.load(MODEL_PATH)
model_pipeline = model_artifact["pipeline"]

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        saved_metadata = json.load(file)
else:
    saved_metadata = model_artifact.get("metrics", {})

app = FastAPI(
    title="Cybersecurity Resolution Time API",
    description=(
        "Predicts cybersecurity incident resolution time using a "
        "lightweight Random Forest model."
    ),
    version="2.0.0",
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Cybersecurity Resolution Time Prediction API",
        "documentation": "/docs",
        "health_check": "/health",
        "metadata": "/metadata",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_status": "ready",
        "model_loaded": model_pipeline is not None,
        "model_type": "scikit-learn RandomForestRegressor",
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return saved_metadata


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IncidentInput) -> PredictionOutput:
    try:
        input_df = pd.DataFrame([create_input_row(data)])
        predicted_hours = max(
            0.0,
            float(model_pipeline.predict(input_df)[0]),
        )

        return PredictionOutput(
            predicted_resolution_hours=round(predicted_hours, 2),
            predicted_resolution_days=round(predicted_hours / 24, 2),
            model="Random Forest",
            unit="hours",
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(error),
            },
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
