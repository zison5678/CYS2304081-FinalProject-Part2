from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_incidents.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_FEATURES = [
    "country",
    "attack_type",
    "target_industry",
    "attack_source",
    "vulnerability_type",
    "defense_mechanism",
]

NUMERIC_FEATURES = [
    "years_since_2015",
    "log_financial_loss",
    "log_affected_users",
    "loss_per_user_usd",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "resolution_time_hours"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "selector",
                SelectPercentile(
                    score_func=f_regression,
                    percentile=60,
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=8,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    X = dataframe[FEATURE_COLUMNS].copy()
    y = dataframe[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "serving_model": "scikit-learn RandomForestRegressor",
        "n_estimators": 80,
        "max_depth": 8,
        "feature_selection_percentile": 60,
        "source_data": "cleaned_incidents.csv exported from Part I",
    }

    # Refit the serving model using all cleaned Part I records.
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "model_version": "lightweight-v1",
    }

    joblib.dump(
        artifact,
        MODEL_DIR / "lightweight_resolution_model.joblib",
        compress=3,
    )

    with open(
        MODEL_DIR / "lightweight_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
