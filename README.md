# Cybersecurity Incident Resolution Time Prediction

This project provides a FastAPI endpoint for predicting cybersecurity incident resolution time and a Streamlit application for visualising cybersecurity data and submitting prediction requests.

## Live FastAPI Deployment

The FastAPI service is deployed on Render:

**API URL:**  
https://cybersecurity-resolution-api-mgvw.onrender.com

**API Documentation:**  
https://cybersecurity-resolution-api-mgvw.onrender.com/docs

**Health Check:**  
https://cybersecurity-resolution-api-mgvw.onrender.com/health

The main prediction endpoint is:

```text
POST /predict

```

## Running the Streamlit Application

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Configure the FastAPI URL

Create the file:

```text
.streamlit/secrets.toml
```

Add:

```toml
API_URL = "https://cybersecurity-resolution-api-mgvw.onrender.com"
```

### 3. Run Streamlit

```bash
streamlit run streamlit_app.py
```

The Streamlit application will normally open at:

```text
http://localhost:8501
```
