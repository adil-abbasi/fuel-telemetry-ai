import streamlit as st


def show_metrics(df):

    # ======================================================
    # GENERATOR METRICS
    # ======================================================

    total_generators = (
        df["generator_id"]
        .nunique()
    )

    running_generators = (
        df[
            df["estimated_status"]
            .fillna("")
            .str.contains(
                "Running",
                case=False,
                na=False
            )
        ]["generator_id"]
        .nunique()
    )

    stopped_generators = (
        total_generators
        - running_generators
    )

    # ======================================================
    # FUEL METRICS
    # ======================================================

    average_fuel = round(
        df["fuel_level_l"]
        .mean(),
        2
    )

    # ======================================================
    # HEALTH METRICS
    # ======================================================

    average_health = round(
        df["overall_sensor_health"]
        .mean(),
        2
    )

    # ======================================================
    # MAINTENANCE ALERTS
    # ======================================================

    alert_priorities = [
        "Medium",
        "High",
        "Critical",
    ]

    maintenance_alerts = (
        df["maintenance_priority"]
        .isin(alert_priorities)
        .sum()
    )

    # ======================================================
    # KPI ROW 1
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Generators",
        total_generators
    )

    c2.metric(
        "Running",
        running_generators
    )

    c3.metric(
        "Stopped",
        stopped_generators
    )

    c4.metric(
        "Average Fuel",
        f"{average_fuel:.1f} L"
    )

    # ======================================================
    # KPI ROW 2
    # ======================================================

    st.write("")

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Sensor Health",
        f"{average_health:.2f}%"
    )

    c6.metric(
        "Maintenance Alerts",
        int(maintenance_alerts)
    )

    c7.metric(
        "Records",
        f"{len(df):,}"
    )