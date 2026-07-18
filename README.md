# Cybersecurity Resolution Time Application

This version follows the Week 11 lecture deployment method:

- `main.py` contains the FastAPI application.
- `model.joblib` is loaded directly at startup.
- `requirements.txt` installs the Python dependencies.
- Render start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

No Docker, Java, or PySpark runtime is required for the deployed API.

## Render settings

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## API routes

- `/`
- `/health`
- `/metadata`
- `/docs`
- `POST /predict`
