# Global Cybersecurity Threat Analytics Application

This project contains a FastAPI prediction endpoint and an interactive Streamlit dashboard built from the cleaned outputs of Part I.

## Deployment model

The Render API uses a lightweight scikit-learn Random Forest serving model trained from `data/cleaned_incidents.csv`, the cleaned dataset exported by Part I. It mirrors the Part I prediction design with the same engineered predictors, 60% univariate feature selection, 80 trees, maximum depth 8, and random seed 42. The original Spark model artefact can remain in the Part I submission for evidence, but it is not loaded by the free Render service because of its Java/Spark memory requirements.

## Live FastAPI

FastAPI endpoint:

`https://cybersecurity-resolution-api-mgvw.onrender.com`

API documentation:

`https://cybersecurity-resolution-api-mgvw.onrender.com/docs`

Health check:

`https://cybersecurity-resolution-api-mgvw.onrender.com/health`

## Run FastAPI locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Run Streamlit locally

Keep FastAPI running and open a second terminal:

```bash
python -m streamlit run streamlit_app.py
```

For Streamlit Community Cloud, set this secret:

```toml
API_URL = "https://cybersecurity-resolution-api-mgvw.onrender.com"
```

## Main files

- `app.py`: lightweight FastAPI service
- `streamlit_app.py`: interactive dashboard and prediction form
- `model/lightweight_resolution_model.joblib`: deployment model
- `train_lightweight_model.py`: reproducible model training script
- `data/`: cleaned Part I outputs used by the dashboard and training script
- `Dockerfile` and `render.yaml`: Render deployment configuration
