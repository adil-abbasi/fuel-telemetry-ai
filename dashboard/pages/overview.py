import streamlit as st
import plotly.express as px

from components.metrics import show_metrics


def show_overview(df):

    # ======================================================
    # HEADER
    # ======================================================

    st.title(
        "Fuel Telemetry AI"
    )

    st.caption(
        "Generator Monitoring & Intelligence Dashboard"
    )

    # ======================================================
    # FILTERS
    # ======================================================

    st.sidebar.subheader(
        "Dashboard Filters"
    )

    generators = sorted(
        df["generator_id"]
        .dropna()
        .unique()
    )

    selected_generator = st.sidebar.selectbox(
        "Generator",
        ["All"] + list(generators)
    )

    filtered_df = df.copy()

    if selected_generator != "All":

        filtered_df = filtered_df[
            filtered_df["generator_id"]
            == selected_generator
        ]

    # ======================================================
    # KPI METRICS
    # ======================================================

    show_metrics(
        filtered_df
    )

    st.divider()

    # ======================================================
    # GENERATOR STATUS
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Generator Status"
        )

        status = (
            filtered_df[
                "estimated_status"
            ]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        status.columns = [
            "Status",
            "Count"
        ]

        fig = px.bar(
            status,
            x="Status",
            y="Count",
            title="Estimated Generator States",
        )

        fig.update_layout(
            xaxis_title="Status",
            yaxis_title="Records",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # ======================================================
    # SENSOR HEALTH
    # ======================================================

    with col2:

        st.subheader(
            "Sensor Health"
        )

        fig2 = px.histogram(
            filtered_df,
            x="overall_sensor_health",
            nbins=20,
            title="Overall Sensor Health",
        )

        fig2.update_layout(
            xaxis_title="Health (%)",
            yaxis_title="Records",
        )

        st.plotly_chart(
            fig2,
            width="stretch",
        )

    # ======================================================
    # FUEL OVERVIEW
    # ======================================================

    st.divider()

    st.subheader(
        "Fuel Level Overview"
    )

    fuel_fig = px.line(
        filtered_df,
        x="timestamp",
        y="fuel_level_l",
        color=(
            "generator_id"
            if selected_generator == "All"
            else None
        ),
        title="Fuel Level Over Time",
    )

    fuel_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Fuel (L)",
    )

    st.plotly_chart(
        fuel_fig,
        width="stretch",
    )

    # ======================================================
    # MAINTENANCE PRIORITY
    # ======================================================

    st.subheader(
        "Maintenance Priority"
    )

    priority = (
        filtered_df[
            "maintenance_priority"
        ]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    priority.columns = [
        "Priority",
        "Count"
    ]

    priority_fig = px.bar(
        priority,
        x="Priority",
        y="Count",
        title="Maintenance Priority Distribution",
    )

    st.plotly_chart(
        priority_fig,
        width="stretch",
    )