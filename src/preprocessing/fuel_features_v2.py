"""
===========================================================
Fuel Feature Engineering V2
===========================================================

Creates ML-ready features for:

• Fuel Forecasting
• Fuel Consumption Prediction
• Generator Behaviour Analysis
• Anomaly Detection

Author: Adil Abbasi
===========================================================
"""

from __future__ import annotations

from click import group
import numpy as np
import pandas as pd


class FuelFeatureEngineerV2:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate(self):

        required = [

            "generator_id",
            "timestamp",

            "fuel_level_l",
            "current",
            "battery_voltage",

            "telemetry_quality_score",
            "imputation_confidence"

        ]

        missing = [

            c for c in required
            if c not in self.df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns: {missing}"

            )

    # --------------------------------------------------
    # Prepare Data
    # --------------------------------------------------

    def prepare(self):

        self.df["timestamp"] = pd.to_datetime(

            self.df["timestamp"]

        )

        self.df = self.df.sort_values(

            [

                "generator_id",
                "timestamp"

            ]

        ).reset_index(drop=True)

        self.group = self.df.groupby(

            "generator_id"

        )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):

        self.validate()

        self.prepare()

        self.create_lag_features()

        self.create_rolling_features()

        self.create_consumption_features()

        self.create_efficiency_features()

        self.create_generator_features()

        self.create_quality_features()

        return self.df     
s# --------------------------------------------------
# Lag Features
# --------------------------------------------------

def create_lag_features(self):

    print("Creating lag features...")

    lag_columns = [

        "fuel_level_l",
        "current",
        "battery_voltage"

    ]

    lags = [

        1,
        5,
        15,
        30,
        60

    ]

    for column in lag_columns:

        for lag in lags:

            self.df[f"{column}_lag_{lag}"] = (

                self.group[column]
                .shift(lag)

            )

# --------------------------------------------------
# Rolling Statistics
# --------------------------------------------------

def create_rolling_features(self):

    print("Creating rolling statistics...")

    windows = [

        5,
        15,
        30,
        60

    ]

    sensors = [

        "fuel_level_l",
        "current",
        "battery_voltage"

    ]

    for sensor in sensors:

        for window in windows:

            rolling = (

                self.group[sensor]
                .rolling(window)

            )

            self.df[
                f"{sensor}_mean_{window}"
            ] = (

                rolling
                .mean()
                .reset_index(level=0, drop=True)

            )

            self.df[
                f"{sensor}_std_{window}"
            ] = (

                rolling
                .std()
                .reset_index(level=0, drop=True)

            )

            self.df[
                f"{sensor}_min_{window}"
            ] = (

                rolling
                .min()
                .reset_index(level=0, drop=True)

            )

            self.df[
                f"{sensor}_max_{window}"
            ] = (

                rolling
                .max()
                .reset_index(level=0, drop=True)

            )

    # --------------------------------------------------
    # Consumption Features
    # --------------------------------------------------

    def create_consumption_features(self):

        print("Creating consumption features...")

        df = self.df
        group = self.group

        # Fuel consumption per minute
        df["fuel_consumption_rate"] = (
            -group["fuel_level_l"].diff()
        )

        df["fuel_consumption_rate"] = (
            df["fuel_consumption_rate"]
            .clip(lower=0)
            .fillna(0)
        )

        # cumulative usage
        df["fuel_used_today"] = (
            group["fuel_consumption_rate"]
            .cumsum()
        )

        self.df = df
    def create_efficiency_features(self):

        print("Creating efficiency features...")

    df = self.df

    df["fuel_per_amp"] = np.where(
        df["current"] > 0,
        df["fuel_consumption_rate"] / df["current"],
        0
    )

    df["fuel_per_hour"] = (
        df["fuel_consumption_rate"] * 60
    )

    self.df = df

def create_generator_features(self):

    print("Creating generator features...")

    df = self.df

    df["generator_running"] = (
        df["status"]
        .eq("running")
        .astype(int)
    )

    df["generator_stopped"] = (
        df["status"]
        .eq("stopped")
        .astype(int)
    )

    df["has_current"] = (
        df["current"] > 5
    ).astype(int)

    df["battery_charging"] = (
        df["battery_voltage"] > 48
    ).astype(int)

    self.df = df
    def create_quality_features(self):

     print("Creating quality features...")

    df = self.df

    if "telemetry_quality_score" in df.columns:

        df["poor_quality"] = (
            df["telemetry_quality_score"] < 80
        ).astype(int)

    else:

        df["poor_quality"] = 0

    if "status_conflict" in df.columns:

        df["status_conflict"] = (
            df["status_conflict"]
            .fillna(False)
            .astype(int)
        )

    self.df = df