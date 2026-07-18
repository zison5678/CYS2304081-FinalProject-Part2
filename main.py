import json
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Limit CPU and memory usage before importing NumPy/scikit-learn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"


# ============================================================
# Application state
# ============================================================

model_pipeline: Any = None
pandas_module: Any = None
saved_metadata: dict[str, Any] = {}

model_status = "not_started"
model_error: Optional[str] = None

model_state_lock = threading.Lock()
prediction_lock = threading.Lock()


# ============================================================
# API input and output
# ============================================================

class IncidentInput(BaseModel):
    country: str = Field(min_length=1)

    year: int = Field(
        ge=2015,
        le=2030
    )

    attack_type: str = Field(min_length=1)
    target_industry: str = Field(min_length=1)

    financial_loss_million: float = Field(
        ge=0
    )

    affected_users: int = Field(
        ge=0
    )

    attack_source: str = Field(min_length=1)
    vulnerability_type: str = Field(min_length=1)
    defense_mechanism: str = Field(min_length=1)


class PredictionOutput(BaseModel):
    predicted_resolution_hours: float
    predicted_resolution_days: float
    model: str
    unit: str


# ============================================================
# Feature engineering
# ============================================================

def create_input_row(
    data: IncidentInput
) -> dict[str, Any]:

    years_since_2015 = data.year - 2015

    log_financial_loss = math.log1p(
        data.financial_loss_million
    )

    log_affected_users = math.log1p(
        data.affected_users
    )

    if data.affected_users > 0:
        loss_per_user_usd = (
            data.financial_loss_million
            * 1_000_000
        ) / data.affected_users
    else:
        loss_per_user_usd = 0.0

    return {
        "country": data.country.strip(),

        "attack_type": (
            data.attack_type.strip()
        ),

        "target_industry": (
            data.target_industry.strip()
        ),

        "attack_source": (
            data.attack_source.strip()
        ),

        "vulnerability_type": (
            data.vulnerability_type.strip()
        ),

        "defense_mechanism": (
            data.defense_mechanism.strip()
        ),

        "years_since_2015": int(
            years_since_2015
        ),

        "log_financial_loss": float(
            log_financial_loss
        ),

        "log_affected_users": float(
            log_affected_users
        ),

        "loss_per_user_usd": float(
            loss_per_user_usd
        )
    }


# ============================================================
# Metadata loading
# ============================================================

def load_metadata_file() -> None:

    global saved_metadata

    if not METADATA_PATH.exists():
        saved_metadata = {}
        return

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            saved_metadata = json.load(file)

    except Exception as error:
        saved_metadata = {
            "metadata_error": str(error)
        }


# ============================================================
# Background model loading
# ============================================================

def load_model_background() -> None:

    global model_pipeline
    global pandas_module
    global saved_metadata
    global model_status
    global model_error

    model_status = "loading_model"
    model_error = None

    print(
        "Loading lightweight model in background...",
        flush=True
    )

    try:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found: {MODEL_PATH}"
            )

        # Heavy libraries are imported here, after FastAPI starts.
        import joblib
        import pandas as pd

        model_artifact = joblib.load(
            MODEL_PATH
        )

        if not isinstance(model_artifact, dict):
            raise RuntimeError(
                "The model.joblib file has an invalid format."
            )

        loaded_pipeline = model_artifact.get(
            "pipeline"
        )

        if loaded_pipeline is None:
            raise RuntimeError(
                "The pipeline was not found inside model.joblib."
            )

        artifact_metrics = model_artifact.get(
            "metrics",
            {}
        )

        with model_state_lock:
            model_pipeline = loaded_pipeline
            pandas_module = pd

            if not saved_metadata:
                saved_metadata = artifact_metrics

            model_status = "ready"
            model_error = None

        print(
            "Lightweight model loaded successfully.",
            flush=True
        )

    except Exception as error:

        with model_state_lock:
            model_pipeline = None
            pandas_module = None
            model_status = "failed"
            model_error = str(error)

        print(
            "Model loading failed:",
            model_error,
            flush=True
        )


# ============================================================
# FastAPI startup
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    global model_status

    load_metadata_file()

    model_status = "waiting_to_load"

    # Wait one second so Uvicorn can open the Render port first.
    loading_thread = threading.Timer(
        1.0,
        load_model_background
    )

    loading_thread.daemon = True
    loading_thread.start()

    print(
        "FastAPI started. Model will load in background.",
        flush=True
    )

    yield


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Cybersecurity Resolution Time API",

    description=(
        "Predicts cybersecurity incident resolution time "
        "using a lightweight Random Forest model."
    ),

    version="2.1.0",

    lifespan=lifespan
)


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {
        "message": (
            "Cybersecurity Resolution Time "
            "Prediction API"
        ),

        "documentation": "/docs",
        "health_check": "/health",
        "metadata": "/metadata",
        "prediction_endpoint": "/predict",
        "model_status": model_status
    }


@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_loaded": model_pipeline is not None,
        "model_error": model_error,
        "model_type": (
            "scikit-learn RandomForestRegressor"
        )
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:

    return saved_metadata


@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: IncidentInput
) -> PredictionOutput:

    if model_status != "ready":
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Prediction model is still loading."
                ),
                "model_status": model_status,
                "model_error": model_error
            }
        )

    with model_state_lock:
        active_pipeline = model_pipeline
        active_pandas = pandas_module

    if (
        active_pipeline is None
        or active_pandas is None
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Prediction model is unavailable."
                ),
                "model_status": model_status,
                "model_error": model_error
            }
        )

    try:
        input_row = create_input_row(
            data
        )

        input_df = active_pandas.DataFrame(
            [input_row]
        )

        with prediction_lock:
            prediction = active_pipeline.predict(
                input_df
            )

        predicted_hours = max(
            0.0,
            float(prediction[0])
        )

        return PredictionOutput(
            predicted_resolution_hours=round(
                predicted_hours,
                2
            ),

            predicted_resolution_days=round(
                predicted_hours / 24,
                2
            ),

            model="Random Forest",
            unit="hours"
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(error)
            }
        ) from error


# ============================================================
# Local execution
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=False,
        workers=1
    )
