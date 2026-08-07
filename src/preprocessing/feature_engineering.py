"""
==============================================================
Feature Engineering
==============================================================

Creates ML-ready features from cleaned telemetry.

Part 1
------
✔ Time features
✔ Delta features
✔ Time interval features

Fuel Telemetry AI Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate(self):

        required = [
            "generator_id",
            "timestamp",
            "fuel_level_l",
            "current",
            "battery_voltage"
        ]

        missing = [
            c for c in required
            if c not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    # ==========================================================
    # Sort Data
    # ==========================================================

    def _prepare(self):

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"]
        )

        self.df = self.df.sort_values(
            [
                "generator_id",
                "timestamp"
            ]
        ).reset_index(drop=True)

    # ==========================================================
    # Time Features
    # ==========================================================

    def _time_features(self):

        ts = self.df["timestamp"]

        self.df["hour"] = ts.dt.hour

        self.df["minute"] = ts.dt.minute

        self.df["day"] = ts.dt.day

        self.df["weekday"] = ts.dt.weekday

        self.df["month"] = ts.dt.month

        self.df["is_weekend"] = (
            ts.dt.weekday >= 5
        ).astype(int)

    # ==========================================================
    # Time Difference
    # ==========================================================

    def _time_delta(self):

        self.df["time_delta_sec"] = (

            self.df

            .groupby("generator_id")["timestamp"]

            .diff()

            .dt.total_seconds()

        )

    # ==========================================================
    # Delta Features
    # ==========================================================

    def _delta_features(self):

        grouped = self.df.groupby(
            "generator_id"
        )

        self.df["fuel_delta"] = (

            grouped["fuel_level_l"]

            .diff()

        )

        self.df["current_delta"] = (

            grouped["current"]

            .diff()

        )

        self.df["voltage_delta"] = (

            grouped["battery_voltage"]

            .diff()

        )

    # ==========================================================
    # Fuel Consumption Rate
    # ==========================================================

    def _fuel_rate(self):

        self.df["fuel_rate_lps"] = (

            -self.df["fuel_delta"]

            /

            self.df["time_delta_sec"]

        )

        self.df["fuel_rate_lph"] = (

            self.df["fuel_rate_lps"] * 3600

        )

    # ==========================================================
    # Cleaning
    # ==========================================================

    def _clean(self):

        numeric = [

            "fuel_delta",

            "current_delta",

            "voltage_delta",

            "fuel_rate_lps",

            "fuel_rate_lph"

        ]

        for col in numeric:

            self.df[col] = (

                self.df[col]

                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )

            )

    # ==========================================================
    # Pipeline
    # ==========================================================

    def run(self):

        self._validate()

        self._prepare()

        self._time_features()

        self._time_delta()

        self._delta_features()

        self._fuel_rate()

        self._clean()

        return self.df