# Global Cybersecurity Threat Analytics Application

## Project Overview

This application analyses global cybersecurity threat incidents from 2015 to 2024.

The project contains:

- An interactive Streamlit dashboard
- Cyber-risk profile visualisation
- A FastAPI prediction service
- A PySpark Random Forest model
- Resolution-time prediction for new cybersecurity incidents

## Project Structure

```text
.
├── app.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── data
│   ├── cleaned_incidents.csv
│   ├── clustered_incidents.csv
│   ├── cluster_profiles.csv
│   └── model_metadata.json
└── model
    └── resolution_time_rf_pipeline_model
        ├── metadata
        └── stages