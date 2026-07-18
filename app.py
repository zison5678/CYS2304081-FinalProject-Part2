import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "model" / "resolution_time_rf_pipeline_model"
    )
)

METADATA_PATH = BASE_DIR / "data" / "model_metadata.json"

spark = None
model = None
metadata = {}


class IncidentInput(BaseModel):
    country: str
    year: int = Field(ge=2015, le=2030)
    attack_type: str
    target_industry: str
    financial_loss_million: float = Field(ge=0)
    affected_users: int = Field(ge=0)
    attack_source: str
    vulnerability_type: str
    defense_mechanism: str


def create_input_row(data: IncidentInput) -> dict:
    years_since_2015 = data.year - 2015

    log_financial_loss = math.log1p(
        data.financial_loss_million
    )

    log_affected_users = math.log1p(
        data.affected_users
    )

    if data.affected_users > 0:
        loss_per_user_usd = (
            data.financial_loss_million * 1_000_000
        ) / data.affected_users
    else:
        loss_per_user_usd = 0.0

    return {
        "country": data.country,
        "year": int(data.year),
        "years_since_2015": int(years_since_2015),

        "attack_type": data.attack_type,
        "target_industry": data.target_industry,

        "financial_loss_million": float(
            data.financial_loss_million
        ),
        "log_financial_loss": float(
            log_financial_loss
        ),

        "affected_users": int(
            data.affected_users
        ),
        "log_affected_users": float(
            log_affected_users
        ),

        "loss_per_user_usd": float(
            loss_per_user_usd
        ),

        "attack_source": data.attack_source,
        "vulnerability_type": data.vulnerability_type,
        "defense_mechanism": data.defense_mechanism
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark, model, metadata

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model folder not found: {MODEL_PATH}"
        )

    spark = (
        SparkSession.builder
        .appName("CyberThreatResolutionAPI")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    model = PipelineModel.load(
        str(MODEL_PATH)
    )

    if METADATA_PATH.exists():
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

    yield

    if spark is not None:
        spark.stop()


app = FastAPI(
    title="Cybersecurity Resolution Time API",
    description=(
        "Predicts incident resolution time using "
        "a Spark Random Forest pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Cybersecurity Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "spark_loaded": spark is not None,
        "model_loaded": model is not None
    }


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
def predict(data: IncidentInput):
    if spark is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not ready."
        )

    try:
        input_row = create_input_row(data)

        input_df = spark.createDataFrame(
            [input_row]
        )

        prediction_row = (
            model.transform(input_df)
            .select("prediction")
            .first()
        )

        predicted_hours = max(
            0.0,
            float(prediction_row["prediction"])
        )

        return {
            "predicted_resolution_hours": round(
                predicted_hours,
                2
            ),
            "predicted_resolution_days": round(
                predicted_hours / 24,
                2
            ),
            "model": "Random Forest",
            "unit": "hours"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )