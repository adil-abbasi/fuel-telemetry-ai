import streamlit as st
import plotly.express as px


def show_fuel(df):

    # ======================================================
    # HEADER
    # ======================================================

    st.title(
        "Fuel Analytics"
    )

    st.caption(
        "Monitor fuel levels, consumption rates, and fuel data quality."
    )

    # ======================================================
    # SIDEBAR FILTER
    # ======================================================

    st.sidebar.subheader(
        "Fuel Filters"
    )

    generators = sorted(
        df["generator_id"]
        .dropna()
        .unique()
    )

    selected = st.sidebar.selectbox(
        "Select Generator",
        ["All"] + list(generators)
    )

    # ------------------------------------------------------
    # Apply Generator Filter
    # ------------------------------------------------------

    filtered_df = df.copy()

    if selected != "All":

        filtered_df = filtered_df[
            filtered_df["generator_id"]
            == selected
        ]

    # ======================================================
    # KPI CARDS
    # ======================================================

    fuel_values = (
        filtered_df["fuel_level_l"]
        .dropna()
    )

    valid_fuel_values = fuel_values[
        fuel_values >= 0
    ]

    avg_fuel = (
        fuel_values.mean()
        if not fuel_values.empty
        else 0
    )

    min_valid_fuel = (
        valid_fuel_values.min()
        if not valid_fuel_values.empty
        else 0
    )

    max_fuel = (
        fuel_values.max()
        if not fuel_values.empty
        else 0
    )

    running_df = filtered_df[
        filtered_df["estimated_status"]
        .fillna("")
        .str.contains(
            "Running",
            case=False,
            na=False
        )
    ]

    running_rates = (
        running_df["fuel_rate_lph"]
        .dropna()
    )

    avg_rate = (
        running_rates.mean()
        if not running_rates.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Fuel",
        f"{avg_fuel:.2f} L"
    )

    c2.metric(
        "Minimum Valid Fuel",
        f"{min_valid_fuel:.2f} L"
    )

    c3.metric(
        "Maximum Fuel",
        f"{max_fuel:.2f} L"
    )

    c4.metric(
        "Running Consumption",
        f"{avg_rate:.2f} L/hr"
    )
    st.divider()

    # ======================================================
    # FUEL LEVEL TREND
    # ======================================================

    st.subheader(
        "Fuel Level Trend"
    )

    if not filtered_df.empty:

        fig = px.line(
            filtered_df,
            x="timestamp",
            y="fuel_level_l",
            color=(
                "generator_id"
                if selected == "All"
                else None
            ),
            title="Fuel Level Over Time",
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Fuel Level (L)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    else:

        st.info(
            "No fuel data available for the selected generator."
        )

    # ======================================================
    # FUEL CONSUMPTION RATE
    # ======================================================

    st.subheader(
        "Fuel Consumption Rate"
    )

    if not filtered_df.empty:

        fig2 = px.line(
            filtered_df,
            x="timestamp",
            y="fuel_rate_lph",
            color=(
                "generator_id"
                if selected == "All"
                else None
            ),
            title="Fuel Consumption Rate Over Time",
        )

        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Fuel Consumption (L/hr)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig2,
            width="stretch",
        )

    else:

        st.info(
            "No consumption data available."
        )

    # ======================================================
    # GENERATOR COMPARISON
    # ======================================================

    st.subheader(
        "Average Fuel Consumption by Generator"
    )

    comparison = (
        filtered_df
        .groupby("generator_id")[
            "fuel_rate_lph"
        ]
        .mean()
        .reset_index()
        .sort_values(
            "fuel_rate_lph",
            ascending=False
        )
    )

    if not comparison.empty:

        fig3 = px.bar(
            comparison,
            x="generator_id",
            y="fuel_rate_lph",
            title="Average Consumption",
        )

        fig3.update_layout(
            xaxis_title="Generator",
            yaxis_title="Average Consumption (L/hr)",
        )

        st.plotly_chart(
            fig3,
            width="stretch",
        )

    else:

        st.info(
            "No generator consumption data available."
        )

    # ======================================================
    # FUEL DATA QUALITY
    # ======================================================

    st.subheader(
        "Fuel Data Quality"
    )

    invalid = (
        filtered_df["fuel_invalid"].sum()
        if "fuel_invalid" in filtered_df.columns
        else 0
    )

    outliers = (
        filtered_df["fuel_outlier"].sum()
        if "fuel_outlier" in filtered_df.columns
        else 0
    )

    imputed = (
        filtered_df["fuel_imputed"].sum()
        if "fuel_imputed" in filtered_df.columns
        else 0
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Invalid Readings",
        f"{int(invalid):,}"
    )

    q2.metric(
        "Outliers",
        f"{int(outliers):,}"
    )

    q3.metric(
        "Imputed Readings",
        f"{int(imputed):,}"
    )

    # ======================================================
    # GENERATOR MINIMUM DIAGNOSTIC
    # ======================================================

    if selected == "All":

        st.divider()

        st.subheader(
            "Minimum Fuel by Generator"
        )

        generator_minimums = (
            filtered_df
            .groupby("generator_id")[
                "fuel_level_l"
            ]
            .min()
            .reset_index()
            .sort_values(
                "fuel_level_l"
            )
        )

        generator_minimums.columns = [
            "Generator",
            "Minimum Fuel (L)"
        ]

        st.dataframe(
            generator_minimums,
            width="stretch",
            hide_index=True,
        )