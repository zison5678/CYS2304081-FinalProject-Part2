import json
import math
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("PYSPARK_PYTHON", "python")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            BASE_DIR
            / "model"
            / "resolution_time_rf_pipeline_model"
        )
    )
)

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "model_metadata.json"
)

spark: Optional[SparkSession] = None
model: Optional[PipelineModel] = None
metadata: dict = {}

model_status = "not_started"
model_error: Optional[str] = None

prediction_lock = threading.Lock()

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

def create_input_row(
    data: IncidentInput
) -> dict:
    """
    Create the raw and engineered columns expected by the
    Spark prediction pipeline.
    """

    years_since_2015 = (
        data.year - 2015
    )

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

        "year": int(
            data.year
        ),

        "years_since_2015": int(
            years_since_2015
        ),

        "attack_type": (
            data.attack_type.strip()
        ),

        "target_industry": (
            data.target_industry.strip()
        ),

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

        "attack_source": (
            data.attack_source.strip()
        ),

        "vulnerability_type": (
            data.vulnerability_type.strip()
        ),

        "defense_mechanism": (
            data.defense_mechanism.strip()
        )
    }

def repair_random_forest_metadata(
    spark_session: SparkSession,
    model_path: Path
) -> None:
    """
    Repairs Spark Random Forest treesMetadata parquet files
    that were exported with generic tuple column names:

        _1, _2, _3

    Spark expects:

        treeID, metadata, weights
    """

    metadata_folders = [
        folder
        for folder in model_path.rglob(
            "treesMetadata"
        )
        if folder.is_dir()
    ]

    if not metadata_folders:
        print(
            "No treesMetadata folder was found. "
            "No metadata repair was required.",
            flush=True
        )
        return

    for metadata_folder in metadata_folders:
        print(
            "Checking Random Forest metadata:",
            metadata_folder,
            flush=True
        )

        metadata_df = (
            spark_session
            .read
            .parquet(
                str(metadata_folder)
            )
        )

        existing_columns = (
            metadata_df.columns
        )

        print(
            "Existing metadata columns:",
            existing_columns,
            flush=True
        )

        expected_columns = [
            "treeID",
            "metadata",
            "weights"
        ]

        if existing_columns == expected_columns:
            print(
                "Random Forest metadata is already correct.",
                flush=True
            )
            continue

        generic_columns = [
            "_1",
            "_2",
            "_3"
        ]

        if existing_columns != generic_columns:
            raise RuntimeError(
                "Unexpected Random Forest metadata columns. "
                f"Found: {existing_columns}. "
                f"Expected either {generic_columns} "
                f"or {expected_columns}."
            )

        temporary_folder = (
            metadata_folder.parent
            / "treesMetadata_repaired"
        )

        if temporary_folder.exists():
            shutil.rmtree(
                temporary_folder
            )

        repaired_df = metadata_df.select(
            F.col("_1")
            .cast("long")
            .alias("treeID"),

            F.col("_2")
            .cast("string")
            .alias("metadata"),

            F.col("_3")
            .cast("double")
            .alias("weights")
        )

        (
            repaired_df
            .coalesce(1)
            .write
            .mode("overwrite")
            .parquet(
                str(temporary_folder)
            )
        )

        shutil.rmtree(
            metadata_folder
        )

        shutil.move(
            str(temporary_folder),
            str(metadata_folder)
        )

        verification_df = (
            spark_session
            .read
            .parquet(
                str(metadata_folder)
            )
        )

        if (
            verification_df.columns
            != expected_columns
        ):
            raise RuntimeError(
                "Random Forest metadata repair failed. "
                f"Columns after repair: "
                f"{verification_df.columns}"
            )

        print(
            "Random Forest metadata repaired successfully:",
            metadata_folder,
            flush=True
        )

def load_application_metadata() -> None:
    global metadata

    if not METADATA_PATH.exists():
        print(
            "Metadata JSON was not found:",
            METADATA_PATH,
            flush=True
        )
        metadata = {}
        return

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        print(
            "Application metadata loaded successfully.",
            flush=True
        )

    except Exception as error:
        metadata = {}

        print(
            "Unable to load application metadata:",
            str(error),
            flush=True
        )

def load_spark_model() -> None:
    global spark
    global model
    global model_status
    global model_error

    model_status = "checking_files"
    model_error = None

    try:
        print(
            "Model path:",
            MODEL_PATH,
            flush=True
        )

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model folder not found: {MODEL_PATH}"
            )

        model_metadata_folder = (
            MODEL_PATH
            / "metadata"
        )

        model_stages_folder = (
            MODEL_PATH
            / "stages"
        )

        if not model_metadata_folder.exists():
            raise RuntimeError(
                "The model metadata folder is missing: "
                f"{model_metadata_folder}"
            )

        if not model_stages_folder.exists():
            raise RuntimeError(
                "The model stages folder is missing: "
                f"{model_stages_folder}"
            )

        model_status = "starting_spark"

        print(
            "Starting Spark...",
            flush=True
        )

        spark = (
            SparkSession.builder
            .appName(
                "CyberThreatResolutionAPI"
            )
            .master("local[1]")
            .config(
                "spark.ui.enabled",
                "false"
            )
            .config(
                "spark.sql.shuffle.partitions",
                "1"
            )
            .config(
                "spark.default.parallelism",
                "1"
            )
            .config(
                "spark.driver.memory",
                os.getenv(
                    "SPARK_DRIVER_MEMORY",
                    "512m"
                )
            )
            .config(
                "spark.driver.host",
                "127.0.0.1"
            )
            .config(
                "spark.driver.bindAddress",
                "127.0.0.1"
            )
            .config(
                "spark.python.worker.reuse",
                "true"
            )
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel(
            "ERROR"
        )

        print(
            "Spark started successfully.",
            flush=True
        )

        model_status = "repairing_metadata"

        repair_random_forest_metadata(
            spark_session=spark,
            model_path=MODEL_PATH
        )

        model_status = "loading_model"

        print(
            "Loading Spark PipelineModel...",
            flush=True
        )

        model = PipelineModel.load(
            str(MODEL_PATH)
        )

        model_status = "ready"
        model_error = None

        print(
            "Spark model loaded successfully.",
            flush=True
        )

    except Exception as error:
        model = None
        model_status = "failed"
        model_error = str(error)

        print(
            "Spark model loading failed:",
            model_error,
            flush=True
        )

@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    load_application_metadata()

    model_loading_thread = threading.Thread(
        target=load_spark_model,
        name="spark-model-loader",
        daemon=True
    )

    model_loading_thread.start()

    yield

    if spark is not None:
        try:
            spark.stop()

            print(
                "Spark stopped successfully.",
                flush=True
            )

        except Exception as error:
            print(
                "Error while stopping Spark:",
                str(error),
                flush=True
            )

app = FastAPI(
    title=(
        "Cybersecurity Resolution Time API"
    ),
    description=(
        "Predicts cybersecurity incident resolution time "
        "using a PySpark Random Forest pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root() -> dict:
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
def health() -> dict:
    """
    Returns HTTP 200 so Render can verify that the FastAPI
    web server is running, while model_status separately
    reports whether Spark finished loading.
    """

    return {
        "status": "healthy",
        "model_status": model_status,
        "spark_loaded": (
            spark is not None
        ),
        "model_loaded": (
            model is not None
        ),
        "model_error": model_error
    }


@app.get("/metadata")
def get_metadata() -> dict:
    return metadata


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
                    "Prediction model is not ready."
                ),
                "model_status": model_status,
                "model_error": model_error
            }
        )

    if spark is None or model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Spark or the prediction model "
                    "is unavailable."
                ),
                "model_status": model_status,
                "model_error": model_error
            }
        )

    try:
        input_row = create_input_row(
            data
        )
        
        with prediction_lock:
            input_df = spark.createDataFrame(
                [input_row]
            )

            prediction_row = (
                model
                .transform(input_df)
                .select("prediction")
                .first()
            )

        if prediction_row is None:
            raise RuntimeError(
                "The model did not return a prediction."
            )

        predicted_hours = max(
            0.0,
            float(
                prediction_row[
                    "prediction"
                ]
            )
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

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Prediction failed."
                ),
                "error": str(error)
            }
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=False
    )
