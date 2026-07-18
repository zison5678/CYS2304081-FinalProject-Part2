# Cybersecurity Incident Analytics and Resolution Time Prediction System

## Student Information

**Student Name:** Lee Zi Xyuan  
**Student ID:** CYS2304081  
**Module:** Big Data Analytics  
**GitHub Branch:** `main`

---

## 1. Project Overview

This project analyses global cybersecurity incidents from 2015 to 2024 and develops a machine learning system to predict how long a cybersecurity incident may take to resolve.

The project is divided into two main parts:

- **Part I:** Data preparation, feature engineering, incident segmentation and machine learning model development using Apache Spark.
- **Part II:** Deployment of the prediction model using FastAPI, Render and Streamlit.

The dataset used is:

```text
Global_Cybersecurity_Threats_2015-2024.csv
```

Each dataset record represents a cybersecurity incident and contains information such as:

- Country
- Year
- Attack type
- Target industry
- Financial loss
- Number of affected users
- Attack source
- Vulnerability type
- Defence mechanism
- Incident resolution time

---

## 2. Project Objectives

The main objectives of this project are:

1. Load and process cybersecurity incident data using Apache Spark.
2. Clean missing, inconsistent and invalid values.
3. Perform feature selection and feature engineering.
4. Segment incidents into cyber-risk profiles using K-Means clustering.
5. Determine an appropriate number of clusters.
6. Analyse and assign descriptive names to each risk profile.
7. Train a regression model to predict incident resolution time.
8. Evaluate the prediction model using regression metrics.
9. Deploy the prediction model as a public FastAPI service.
10. Build an interactive Streamlit dashboard for data visualisation and prediction.

---

# Part I: Apache Spark Data Science Workflow

## 3. Data Loading and Preparation

Apache Spark was used to process the cybersecurity dataset.

The preparation process included:

- Loading the dataset into a Spark DataFrame
- Displaying sample records
- Inspecting the original schema
- Renaming columns using consistent `snake_case` names
- Correcting numerical and categorical data types
- Trimming inconsistent text values
- Checking missing and null values
- Handling invalid numerical values
- Detecting duplicate records
- Checking numerical distributions
- Treating extreme outliers
- Validating the cleaned dataset

The cleaned dataset was saved for use by the Streamlit application.

---

## 4. Feature Selection

The prediction model uses cybersecurity incident characteristics that may influence resolution time.

The selected raw features include:

```text
country
year
attack_type
target_industry
financial_loss_million
affected_users
attack_source
vulnerability_type
defense_mechanism
```

The target variable is:

```text
resolution_time_hours
```

This target represents the number of hours required to resolve an incident.

---

## 5. Feature Engineering

Several additional features were created to improve the analysis.

### Years Since 2015

```text
years_since_2015 = year - 2015
```

This feature represents the position of each incident within the dataset time period.

### Log Financial Loss

```text
log_financial_loss = log(1 + financial_loss_million)
```

This transformation reduces the effect of very large financial-loss values.

### Log Affected Users

```text
log_affected_users = log(1 + affected_users)
```

This transformation reduces the effect of incidents with extremely large user counts.

### Loss Per User

```text
loss_per_user_usd =
financial_loss_million × 1,000,000 ÷ affected_users
```

This feature estimates the financial impact for each affected user.

Categorical variables were encoded using Spark ML feature transformers before model training.

---

## 6. Cyber-Risk Profile Segmentation

K-Means clustering from Spark MLlib was used to group cybersecurity incidents into different risk profiles.

The clustering process included:

1. Selecting features related to incident impact.
2. Encoding categorical values.
3. Assembling numerical features into feature vectors.
4. Scaling clustering features.
5. Training K-Means models using different values of `K`.
6. Comparing cluster quality using silhouette scores.
7. Reviewing the elbow pattern.
8. Selecting an appropriate number of clusters.
9. Analysing the characteristics of each cluster.
10. Assigning descriptive names to the clusters.

The clusters were analysed using:

- Average financial loss
- Average number of affected users
- Average resolution time
- Common attack type
- Common target industry
- Incident frequency
- Financial and operational impact

The final clustering results were saved in:

```text
data/clustered_incidents.csv
data/cluster_profiles.csv
```

---

## 7. Resolution Time Prediction Model

A Random Forest regression model was developed to predict cybersecurity incident resolution time.

The model receives incident characteristics as input and predicts:

```text
Incident Resolution Time in Hours
```

Random Forest was selected because it can:

- Model non-linear relationships
- Process many input features
- Handle interactions between features
- Reduce overfitting by combining multiple decision trees
- Work effectively with mixed cybersecurity incident characteristics

---

## 8. Model Evaluation

The prediction model was evaluated using the following regression metrics:

### Root Mean Squared Error

```text
RMSE
```

RMSE measures the average prediction error while giving more importance to large errors.

### Mean Absolute Error

```text
MAE
```

MAE measures the average absolute difference between the actual and predicted resolution time.

### Coefficient of Determination

```text
R²
```

R² measures how much variation in resolution time is explained by the model.

A residual analysis was also performed to compare:

```text
Actual Resolution Time
Predicted Resolution Time
Residual Error
```

The evaluation results and limitations are discussed transparently in the Part I notebook.

---

## 9. Part I Outputs

The Part I notebook produces and saves the following outputs:

```text
cleaned_incidents.csv
clustered_incidents.csv
cluster_profiles.csv
model_metadata.json
risk_profile_mapping.json
resolution_time_rf_pipeline_model
cluster_preparation_pipeline_model
kmeans_risk_profile_model
```

These outputs are used by the Part II FastAPI and Streamlit applications.

---

# Part II: FastAPI and Streamlit Application

## 10. Application Architecture

The application follows this workflow:

```text
User
  ↓
Streamlit Dashboard
  ↓
FastAPI Prediction API
  ↓
Random Forest Model
  ↓
Predicted Resolution Time
  ↓
Result Displayed in Hours and Days
```

Apache Spark was used for the main data preparation, clustering and model-development workflow in Part I.

For the public Render deployment, a lightweight deployment-compatible scikit-learn Random Forest model is used. It uses the prepared Part I data and the same incident input logic.

This approach avoids the high Java and PySpark memory requirements that exceed the limitations of the free cloud environment.

---

## 11. FastAPI Backend

The FastAPI backend provides endpoints for checking the system, viewing metadata and requesting predictions.

### Public API URL

```text
https://cybersecurity-resolution-api-mgvw.onrender.com
```

### API Documentation

```text
https://cybersecurity-resolution-api-mgvw.onrender.com/docs
```

### Health Check

```text
https://cybersecurity-resolution-api-mgvw.onrender.com/health
```

### Render Service ID

```text
srv-d9dt7ajrjlhs73baopt0
```

---

## 12. API Health Status

The deployed service returned the following successful health response:

```json
{
  "status": "healthy",
  "model_status": "ready",
  "model_loaded": true,
  "model_error": null,
  "model_type": "scikit-learn RandomForestRegressor"
}
```

This confirms that:

- The FastAPI service is online.
- The model has loaded successfully.
- The prediction endpoint is available.
- No model-loading error is present.

---

## 13. API Endpoints

### Root Endpoint

```http
GET /
```

Returns general information about the API.

### Health Endpoint

```http
GET /health
```

Returns the API and model status.

### Metadata Endpoint

```http
GET /metadata
```

Returns project and model metadata.

### Prediction Endpoint

```http
POST /predict
```

Accepts cybersecurity incident information and returns the predicted resolution time.

---

## 14. Prediction Input Fields

The `/predict` endpoint accepts the following fields:

| Field | Type | Description |
|---|---|---|
| `country` | String | Country where the incident occurred |
| `year` | Integer | Year of the incident |
| `attack_type` | String | Type of cybersecurity attack |
| `target_industry` | String | Industry targeted by the attack |
| `financial_loss_million` | Float | Estimated financial loss in million USD |
| `affected_users` | Integer | Number of affected users |
| `attack_source` | String | Source of the attack |
| `vulnerability_type` | String | Vulnerability used by the attacker |
| `defense_mechanism` | String | Defence mechanism used by the organisation |

---

## 15. Prediction Request Example

The following example represents a ransomware incident affecting the banking industry in China:

```json
{
  "country": "China",
  "year": 2024,
  "attack_type": "Ransomware",
  "target_industry": "Banking",
  "financial_loss_million": 50.0,
  "affected_users": 500000,
  "attack_source": "Hacker Group",
  "vulnerability_type": "Unpatched Software",
  "defense_mechanism": "Firewall"
}
```

---

## 16. Prediction Response Format

A successful request returns a response in this structure:

```json
{
  "predicted_resolution_hours": 36.73,
  "predicted_resolution_days": 1.53,
  "model": "Random Forest",
  "unit": "hours"
}
```

The output contains:

- Predicted resolution time in hours
- Predicted resolution time in days
- Model type
- Prediction unit

For example:

```text
36.73 hours ≈ 1.53 days
```

The prediction is an estimated value produced by the machine learning model.

---

## 17. Testing the API

The API can be tested through Swagger documentation.

Open:

```text
https://cybersecurity-resolution-api-mgvw.onrender.com/docs
```

Then:

1. Open `POST /predict`.
2. Click **Try it out**.
3. Enter the incident information.
4. Click **Execute**.
5. Review the response body.

A successful prediction should return:

```text
HTTP Status Code: 200
```

---

## 18. Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for exploring cybersecurity incidents.

The dashboard contains three main sections:

### Dashboard

The main dashboard displays:

- Total number of incidents
- Total financial loss
- Average incident resolution time
- Number of identified risk profiles
- Financial loss over time
- Attack-type distribution
- Cybersecurity incidents by year

### Risk Profile Analysis

The risk-profile section displays:

- Risk-profile distribution
- Number of incidents in each profile
- Average financial loss
- Average affected users
- Common attack type
- Risk-profile summary table

### Prediction

The prediction section allows users to enter:

- Country
- Year
- Attack type
- Target industry
- Financial loss
- Number of affected users
- Attack source
- Vulnerability type
- Defence mechanism

The Streamlit application sends the information to the FastAPI `/predict` endpoint and displays the predicted resolution time.

---

## 19. GitHub Repository

The project source code is available at:

```text
https://github.com/zison5678/CYS2304081-FinalProject-Part2
```

Git clone command:

```bash
git clone https://github.com/zison5678/CYS2304081-FinalProject-Part2.git
```

---

## 20. Project Structure

```text
CYS2304081-FinalProject-Part2/
│
├── main.py
├── model.joblib
├── model_metadata.json
├── requirements.txt
├── render.yaml
├── streamlit_app.py
├── README.md
│
└── data/
    ├── cleaned_incidents.csv
    ├── clustered_incidents.csv
    ├── cluster_profiles.csv
    └── model_metadata.json
```

### File Descriptions

| File | Purpose |
|---|---|
| `main.py` | FastAPI backend and prediction endpoints |
| `model.joblib` | Lightweight deployed Random Forest model |
| `model_metadata.json` | Model information and evaluation metadata |
| `requirements.txt` | Required Python libraries |
| `render.yaml` | Render deployment configuration |
| `streamlit_app.py` | Interactive Streamlit dashboard |
| `README.md` | Project documentation |
| `cleaned_incidents.csv` | Cleaned cybersecurity dataset |
| `clustered_incidents.csv` | Incidents with assigned risk profiles |
| `cluster_profiles.csv` | Summary of identified risk profiles |

---

## 21. Technologies Used

### Big Data Processing

- Apache Spark
- PySpark
- Spark SQL
- Spark MLlib

### Machine Learning

- K-Means Clustering
- Random Forest Regression
- Scikit-learn
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Dashboard

- Streamlit
- Pandas
- Plotly

### Deployment and Version Control

- Render
- GitHub
- Git

---

## 22. Running the FastAPI Application Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/zison5678/CYS2304081-FinalProject-Part2.git
```

### Step 2: Enter the Project Folder

```bash
cd CYS2304081-FinalProject-Part2
```

### Step 3: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Start FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 5: Open the API Documentation

```text
http://127.0.0.1:8000/docs
```

---

## 23. Running the Streamlit Application Locally

Start the FastAPI backend first.

Then open another terminal and run:

```bash
streamlit run streamlit_app.py
```

The local Streamlit application normally opens at:

```text
http://localhost:8501
```

The local API URL should be:

```text
http://127.0.0.1:8000
```

---

## 24. Streamlit Cloud Configuration

When deploying the Streamlit application, configure the following secret:

```toml
API_URL = "https://cybersecurity-resolution-api-mgvw.onrender.com"
```

This allows the Streamlit prediction form to communicate with the public FastAPI service.

After deployment, add the public Streamlit URL below:

```text
Streamlit Application:
ADD_STREAMLIT_URL_HERE
```

---

## 25. Render Deployment Configuration

The Render Web Service uses the following settings:

### Runtime

```text
Python 3
```

### Python Version

```text
3.11.11
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Instance Type

```text
Free
```

The application binds to:

```text
0.0.0.0
```

and uses Render's assigned:

```text
$PORT
```

The model loads in the background so that the FastAPI server can open the Render port before loading the machine learning libraries.

---

## 26. Model and System Limitations

The project has several limitations:

- The model is trained using historical cybersecurity incident data.
- The dataset may not represent every real-world attack scenario.
- Incident resolution time depends on factors not included in the dataset.
- Organisation size, employee expertise and incident-response readiness may affect the actual resolution time.
- Some attack categories may have fewer records than others.
- Machine learning predictions are estimates and should not be treated as guaranteed values.
- The free Render service has limited computing resources.
- The free service may take longer to respond after a period of inactivity.
- A lightweight model is used for deployment because a complete PySpark environment requires more memory.
- The Streamlit dashboard depends on the availability of the FastAPI backend.

---

## 27. Future Improvements

Possible future improvements include:

- Collecting more recent cybersecurity incident data
- Adding more organisational and technical response features
- Testing additional regression algorithms
- Comparing Random Forest with Gradient Boosting and XGBoost
- Improving the model using hyperparameter tuning
- Adding confidence intervals to predictions
- Implementing automatic incident risk-profile prediction
- Adding authentication to protect the prediction API
- Adding database storage for prediction records
- Deploying the Spark model on a higher-memory cloud service
- Adding real-time cybersecurity incident data

---

## 28. Conclusion

This project demonstrates a complete big-data analytics workflow for cybersecurity incident analysis.

Apache Spark was used to:

- Load and clean the dataset
- Correct data inconsistencies
- Engineer useful features
- Segment incidents into cyber-risk profiles
- Train a Random Forest regression model
- Evaluate model performance
- Save processed datasets and model artefacts

FastAPI was used to create a public prediction service, while Render was used to deploy the API.

The deployed API successfully loads the Random Forest model and returns predicted incident resolution time in hours and days.

Streamlit provides an interactive dashboard for:

- Exploring cybersecurity incident trends
- Comparing attack distributions
- Analysing cyber-risk profiles
- Submitting new incidents for prediction

The project successfully connects big-data processing, machine learning, data visualisation and cloud deployment in one complete cybersecurity analytics application.

---

## 29. Project Links

### GitHub Repository

```text
https://github.com/zison5678/CYS2304081-FinalProject-Part2
```

### Public FastAPI Service

```text
https://cybersecurity-resolution-api-mgvw.onrender.com
```

### FastAPI Documentation

```text
https://cybersecurity-resolution-api-mgvw.onrender.com/docs
```

### FastAPI Health Check

```text
https://cybersecurity-resolution-api-mgvw.onrender.com/health
```

### Streamlit Application

```text
ADD_STREAMLIT_URL_HERE
```

---

## 30. Submission Checklist

- [x] Dataset loaded using Apache Spark
- [x] Data cleaning and preprocessing completed
- [x] Missing and invalid values handled
- [x] Feature selection completed
- [x] Feature engineering completed
- [x] K-Means clustering implemented
- [x] Appropriate cluster number evaluated
- [x] Risk profiles analysed and named
- [x] Random Forest regression model trained
- [x] RMSE, MAE and R² evaluated
- [x] Residual analysis completed
- [x] Cleaned data saved
- [x] Cluster results saved
- [x] Model artefacts saved
- [x] FastAPI backend created
- [x] `/health` endpoint working
- [x] `/metadata` endpoint working
- [x] `/predict` endpoint working
- [x] Model loaded successfully
- [x] FastAPI deployed publicly on Render
- [x] Swagger API documentation available
- [x] Streamlit dashboard code included
- [x] Financial-loss visualisation included
- [x] Attack-distribution visualisation included
- [x] Risk-profile visualisation included
- [x] Prediction form included
- [ ] Public Streamlit URL added

---

**Student:** Lee Zi Xyuan  
**Student ID:** CYS2304081
