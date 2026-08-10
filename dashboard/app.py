import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).parent

sys.path.append(
    str(BASE_DIR)
)

from pages.sensor_health import show_sensor_health
from data_loader import load_data
from pages.overview import show_overview



st.set_page_config(
    page_title="Fuel Telemetry AI",
    page_icon="⚡",
    layout="wide"
)



@st.cache_data
def get_dataset():

    return load_data()



try:

    df = get_dataset()


except Exception as e:

    st.error(
        f"Data loading failed:\n{e}"
    )

    st.stop()



st.sidebar.title(
    "Fuel Telemetry AI"
)

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview",
        "Fuel Analytics",
        "Sensor Health",
        "Anomalies",
    ]
)


if page == "Overview":

    show_overview(df)


elif page == "Fuel Analytics":

    from pages.fuel import show_fuel

    show_fuel(df)


elif page == "Sensor Health":

    show_sensor_health(df)


elif page == "Anomalies":

    st.info(
        "Anomaly Center coming soon."
    )