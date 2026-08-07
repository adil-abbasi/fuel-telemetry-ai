"""
Recalculate engineered features after
validation and imputation.

Author: Adil Abbasi
"""

from __future__ import annotations

import numpy as np
import pandas as pd
class FinalFeatureUpdater:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"]
        )

        self.df = self.df.sort_values(
            ["generator_id", "timestamp"]
        )
    def update_time_features(self):

        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["minute"] = self.df["timestamp"].dt.minute
        self.df["day"] = self.df["timestamp"].dt.day

        self.df["weekday"] = self.df["timestamp"].dt.weekday

        self.df["month"] = self.df["timestamp"].dt.month

        self.df["is_weekend"] = (
            self.df["weekday"] >= 5
        ).astype(int)
    def update_time_delta(self):

        self.df["time_delta_sec"] = (

            self.df
            .groupby("generator_id")["timestamp"]
            .diff()
            .dt.total_seconds()

        )

        self.df["time_delta_sec"] = (
            self.df["time_delta_sec"]
            .fillna(0)
        )
    def update_current_delta(self):

        self.df["current_delta"] = (

            self.df
            .groupby("generator_id")["current"]
            .diff()

        )
    def update_voltage_delta(self):

        self.df["voltage_delta"] = (

            self.df
            .groupby("generator_id")["battery_voltage"]
            .diff()

        )
    def update_voltage_delta(self):

        self.df["voltage_delta"] = (

            self.df
            .groupby("generator_id")["battery_voltage"]
            .diff()

        )
    def update_fuel_rate(self):

        self.df["fuel_rate_lps"] = np.where(

            self.df["time_delta_sec"] > 0,

            -self.df["fuel_delta"] /
            self.df["time_delta_sec"],

            np.nan

        )

        self.df["fuel_rate_lph"] = (

            self.df["fuel_rate_lps"] * 3600

        )
        self.df.loc[
            self.df["fuel_rate_lph"] < 0,
            "fuel_rate_lph"
        ] = np.nan

        self.df.loc[
            self.df["fuel_rate_lps"] < 0,
            "fuel_rate_lps"
        ] = np.nan

    def update_quality_flags(self):

        self.df["fuel_missing"] = (
            self.df["fuel_level_l"].isna()
        )

        self.df["current_missing"] = (
            self.df["current"].isna()
        )

        self.df["battery_missing"] = (
            self.df["battery_voltage"].isna()
        )

        self.df["status_missing"] = (
            self.df["status"].isna()
        )

    def run(self):

        self.update_time_features()

        self.update_time_delta()

        self.update_fuel_delta()

        self.update_current_delta()

        self.update_voltage_delta()

        self.update_fuel_rate()

        self.update_quality_flags()

        return self.df