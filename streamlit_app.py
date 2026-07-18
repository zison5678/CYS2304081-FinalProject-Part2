from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Order of precedence: Streamlit secrets -> env var -> local dev default.
# Set this in .streamlit/secrets.toml (locally) or in the Streamlit Cloud
# "Secrets" panel once the FastAPI service is deployed on Render, e.g.:
#   API_URL = "https://your-service-name.onrender.com"
API_URL = st.secrets.get(
    "API_URL",
    os.getenv("API_URL", "http://127.0.0.1:8000")
)


st.set_page_config(
    page_title="Cybersecurity Threat Analytics",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cybersecurity Threat Analytics")
st.write(
    "Explore cybersecurity incidents, risk profiles, "
    "and predict incident resolution time."
)


@st.cache_data
def load_data():

    cleaned_df = pd.read_csv(
        DATA_DIR / "cleaned_incidents.csv"
    )

    clustered_df = pd.read_csv(
        DATA_DIR / "clustered_incidents.csv"
    )

    profiles_df = pd.read_csv(
        DATA_DIR / "cluster_profiles.csv"
    )

    return cleaned_df, clustered_df, profiles_df


try:

    cleaned_df, clustered_df, profiles_df = load_data()

except FileNotFoundError as error:

    st.error("A required CSV file could not be found.")

    st.code(str(error))

    st.write(
        "Place cleaned_incidents.csv, clustered_incidents.csv "
        "and cluster_profiles.csv inside the data folder."
    )

    st.stop()

except Exception as error:

    st.error(f"Unable to load data: {error}")
    st.stop()


dashboard_tab, risk_tab, prediction_tab = st.tabs(
    [
        "Dashboard",
        "Risk Profiles",
        "Prediction"
    ]
)

with dashboard_tab:

    st.header("Cybersecurity Incident Dashboard")

    total_incidents = len(cleaned_df)

    total_financial_loss = cleaned_df[
        "financial_loss_million"
    ].sum()

    average_resolution_time = cleaned_df[
        "resolution_time_hours"
    ].mean()

    total_risk_profiles = clustered_df[
        "risk_profile_id"
    ].nunique()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Total Incidents",
        f"{total_incidents:,}"
    )

    metric2.metric(
        "Total Financial Loss",
        f"${total_financial_loss:,.2f}M"
    )

    metric3.metric(
        "Average Resolution Time",
        f"{average_resolution_time:.2f} Hours"
    )

    metric4.metric(
        "Risk Profiles",
        total_risk_profiles
    )

    st.divider()

    yearly_loss = (
        cleaned_df
        .groupby(
            "year",
            as_index=False
        )
        .agg(
            total_financial_loss=(
                "financial_loss_million",
                "sum"
            )
        )
    )

    financial_loss_chart = px.line(
        yearly_loss,
        x="year",
        y="total_financial_loss",
        markers=True,
        title="Financial Loss Over Time"
    )

    financial_loss_chart.update_layout(
        xaxis_title="Year",
        yaxis_title="Financial Loss (Million USD)"
    )

    st.plotly_chart(
        financial_loss_chart,
        use_container_width=True
    )

    attack_distribution = (
        cleaned_df["attack_type"]
        .value_counts()
        .reset_index()
    )

    attack_distribution.columns = [
        "attack_type",
        "incident_count"
    ]

    attack_chart = px.bar(
        attack_distribution,
        x="attack_type",
        y="incident_count",
        title="Attack Type Distribution"
    )

    attack_chart.update_layout(
        xaxis_title="Attack Type",
        yaxis_title="Number of Incidents"
    )

    st.plotly_chart(
        attack_chart,
        use_container_width=True
    )

    yearly_incidents = (
        cleaned_df
        .groupby(
            "year",
            as_index=False
        )
        .size()
    )

    yearly_incidents.columns = [
        "year",
        "incident_count"
    ]

    incident_chart = px.line(
        yearly_incidents,
        x="year",
        y="incident_count",
        markers=True,
        title="Cybersecurity Incidents by Year"
    )

    incident_chart.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Incidents"
    )

    st.plotly_chart(
        incident_chart,
        use_container_width=True
    )

with risk_tab:

    st.header("Cyber-Risk Profile Analysis")

    profile_distribution = (
        clustered_df["risk_profile_name"]
        .value_counts()
        .reset_index()
    )

    profile_distribution.columns = [
        "risk_profile_name",
        "incident_count"
    ]

    profile_chart = px.bar(
        profile_distribution,
        x="risk_profile_name",
        y="incident_count",
        title="Risk Profile Distribution"
    )

    profile_chart.update_layout(
        xaxis_title="Risk Profile",
        yaxis_title="Number of Incidents"
    )

    st.plotly_chart(
        profile_chart,
        use_container_width=True
    )

    risk_scatter = px.scatter(
        profiles_df,
        x="avg_financial_loss_million",
        y="avg_affected_users",
        size="incident_count",
        color="common_attack_type",
        hover_name="risk_profile_name",
        title="Financial Loss and User Impact by Risk Profile"
    )

    risk_scatter.update_layout(
        xaxis_title="Average Financial Loss (Million USD)",
        yaxis_title="Average Affected Users"
    )

    st.plotly_chart(
        risk_scatter,
        use_container_width=True
    )

    st.subheader("Risk Profile Summary")

    st.dataframe(
        profiles_df,
        use_container_width=True,
        hide_index=True
    )

with prediction_tab:

    st.header("Predict Incident Resolution Time")

    st.write(
        "Enter the characteristics of a new cybersecurity incident."
    )

    api_url = st.text_input(
        "FastAPI URL",
        value=API_URL
    ).rstrip("/")

    input_column1, input_column2 = st.columns(2)

    with input_column1:

        country = st.selectbox(
            "Country",
            sorted(
                cleaned_df["country"]
                .dropna()
                .unique()
            )
        )

        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=2030,
            value=2024,
            step=1
        )

        attack_type = st.selectbox(
            "Attack Type",
            sorted(
                cleaned_df["attack_type"]
                .dropna()
                .unique()
            )
        )

        target_industry = st.selectbox(
            "Target Industry",
            sorted(
                cleaned_df["target_industry"]
                .dropna()
                .unique()
            )
        )

        financial_loss = st.number_input(
            "Financial Loss (Million USD)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with input_column2:

        affected_users = st.number_input(
            "Number of Affected Users",
            min_value=0,
            value=100000,
            step=1000
        )

        attack_source = st.selectbox(
            "Attack Source",
            sorted(
                cleaned_df["attack_source"]
                .dropna()
                .unique()
            )
        )

        vulnerability_type = st.selectbox(
            "Security Vulnerability Type",
            sorted(
                cleaned_df["vulnerability_type"]
                .dropna()
                .unique()
            )
        )

        defense_mechanism = st.selectbox(
            "Defense Mechanism",
            sorted(
                cleaned_df["defense_mechanism"]
                .dropna()
                .unique()
            )
        )

    if st.button(
        "Predict Resolution Time",
        type="primary",
        use_container_width=True
    ):

        request_data = {
            "country": country,
            "year": int(year),
            "attack_type": attack_type,
            "target_industry": target_industry,
            "financial_loss_million": float(
                financial_loss
            ),
            "affected_users": int(
                affected_users
            ),
            "attack_source": attack_source,
            "vulnerability_type": vulnerability_type,
            "defense_mechanism": defense_mechanism
        }

        try:

            with st.spinner(
                "Sending incident information to FastAPI..."
            ):

                response = requests.post(
                    f"{api_url}/predict",
                    json=request_data,
                    timeout=120
                )

            response.raise_for_status()

            prediction_result = response.json()

            predicted_hours = prediction_result[
                "predicted_resolution_hours"
            ]

            predicted_days = prediction_result[
                "predicted_resolution_days"
            ]

            st.success(
                "Prediction completed successfully."
            )

            result1, result2 = st.columns(2)

            result1.metric(
                "Predicted Resolution Time",
                f"{predicted_hours:.2f} Hours"
            )

            result2.metric(
                "Estimated Resolution Time",
                f"{predicted_days:.2f} Days"
            )

            st.subheader("Submitted Incident Information")

            st.json(request_data)

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to FastAPI. "
                "Make sure app.py is running on port 8000."
            )

        except requests.exceptions.Timeout:

            st.error(
                "FastAPI took too long to respond."
            )

        except requests.exceptions.HTTPError:

            st.error(
                f"FastAPI returned an error: {response.text}"
            )

        except KeyError as error:

            st.error(
                f"The API response is missing this value: {error}"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )