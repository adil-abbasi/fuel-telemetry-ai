import streamlit as st
import plotly.express as px


def show_sensor_health(df):

    st.title(
        "Sensor Health"
    )

    st.caption(
        "Monitor sensor reliability and maintenance conditions across generators."
    )

    # ======================================================
    # SIDEBAR FILTER
    # ======================================================

    st.sidebar.subheader(
        "Sensor Health Filters"
    )

    generators = sorted(
        df["generator_id"]
        .dropna()
        .unique()
    )

    selected = st.sidebar.selectbox(
        "Sensor Health Generator",
        ["All"] + list(generators)
    )

    filtered_df = df.copy()

    if selected != "All":

        filtered_df = filtered_df[
            filtered_df["generator_id"] == selected
        ]

    # ======================================================
    # KPI METRICS
    # ======================================================

    average_health = (
        filtered_df["overall_sensor_health"].mean()
    )

    healthy_records = (
        filtered_df["overall_sensor_health"] >= 90
    ).sum()

    degraded_records = (
        filtered_df["overall_sensor_health"] < 75
    ).sum()

    maintenance_records = (
        filtered_df["maintenance_priority"]
        .isin(
            [
                "Medium",
                "High",
                "Critical",
            ]
        )
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Sensor Health",
        f"{average_health:.2f}%"
    )

    c2.metric(
        "Healthy Records",
        f"{healthy_records:,}"
    )

    c3.metric(
        "Degraded Records",
        f"{degraded_records:,}"
    )

    c4.metric(
        "Maintenance Records",
        f"{maintenance_records:,}"
    )

    st.divider()

    # ======================================================
    # SENSOR HEALTH COMPARISON
    # ======================================================

    st.subheader(
        "Sensor Health Comparison"
    )

    sensor_columns = [
        "fuel_sensor_health",
        "current_sensor_health",
        "battery_sensor_health",
        "status_sensor_health",
    ]

    available_columns = [
        column
        for column in sensor_columns
        if column in filtered_df.columns
    ]

    if available_columns:

        sensor_means = (
            filtered_df[
                available_columns
            ]
            .mean()
            .reset_index()
        )

        sensor_means.columns = [
            "Sensor",
            "Health"
        ]

        sensor_means["Sensor"] = (
            sensor_means["Sensor"]
            .str.replace(
                "_sensor_health",
                "",
                regex=False
            )
            .str.title()
        )

        fig = px.bar(
            sensor_means,
            x="Sensor",
            y="Health",
            title="Average Health by Sensor",
        )

        fig.update_layout(
            xaxis_title="Sensor",
            yaxis_title="Health (%)",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # ======================================================
    # OVERALL HEALTH DISTRIBUTION
    # ======================================================

    st.subheader(
        "Overall Sensor Health Distribution"
    )

    fig2 = px.histogram(
        filtered_df,
        x="overall_sensor_health",
        nbins=20,
        title="Overall Sensor Health",
    )

    fig2.update_layout(
        xaxis_title="Overall Health (%)",
        yaxis_title="Records",
    )

    st.plotly_chart(
        fig2,
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

    fig3 = px.bar(
        priority,
        x="Priority",
        y="Count",
        title="Maintenance Priority Distribution",
    )

    st.plotly_chart(
        fig3,
        width="stretch",
    )

    # ======================================================
    # SENSOR HEALTH TREND
    # ======================================================

    st.subheader(
        "Sensor Health Over Time"
    )

    trend = (
        filtered_df
        .sort_values("timestamp")
    )

    fig4 = px.line(
        trend,
        x="timestamp",
        y="overall_sensor_health",
        color=(
            "generator_id"
            if selected == "All"
            else None
        ),
        title="Overall Sensor Health Trend",
    )

    fig4.update_layout(
        xaxis_title="Time",
        yaxis_title="Health (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
    )

    st.plotly_chart(
        fig4,
        width="stretch",
    )

    # ======================================================
    # MAINTENANCE DETAILS
    # ======================================================

    st.subheader(
        "Maintenance Details"
    )

    detail_columns = [
        "timestamp",
        "generator_id",
        "overall_sensor_health",
        "maintenance_priority",
        "maintenance_reason",
        "recommended_action",
    ]

    existing_columns = [
        column
        for column in detail_columns
        if column in filtered_df.columns
    ]

    details = (
        filtered_df[
            existing_columns
        ]
        .sort_values(
            "overall_sensor_health"
        )
        .head(100)
    )

    st.dataframe(
        details,
        width="stretch",
        hide_index=True,
    )