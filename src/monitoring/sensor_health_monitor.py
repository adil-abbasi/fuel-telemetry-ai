"""
===========================================================
Sensor Health Monitor (MVP)
-----------------------------------------------------------
Purpose:
    Evaluate generator sensor health using simple engineering
    rules. This module is designed for dashboard integration
    and can later be upgraded with statistical and ML models.

Input:
    Feature Engineered Dataset

Output:
    Dataset with sensor health indicators
===========================================================
"""

from __future__ import annotations
from unittest import result

import numpy as np
import pandas as pd


class SensorHealthMonitor:
    """
    Sensor Health Monitor (MVP)

    Performs basic diagnostics on:

    - Fuel Sensor
    - Current Sensor
    - Battery Sensor
    - Status Sensor

    Generates dashboard-ready health scores and
    maintenance recommendations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fuel_capacity: float = 2000.0,
    ):
        self.df = df.copy()

        self.fuel_capacity = fuel_capacity

        # --------------------------------------------------
        # Thresholds
        # --------------------------------------------------

        self.MIN_BATTERY = 0.0
        self.MAX_BATTERY = 60.0

        self.MIN_CURRENT = 0.0
        self.MAX_CURRENT = 400.0

        self.MIN_FUEL = 0.0
        self.MAX_FUEL = fuel_capacity

    # ======================================================
    # Preparation
    # ======================================================

    def prepare(self):
        """
        Prepare dataframe before diagnostics.
        """

        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        self.df = self.df.sort_values(
            ["generator_id", "timestamp"]
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Initialize output columns
        # --------------------------------------------------

        output_columns = [
            "fuel_sensor_health",
            "current_sensor_health",
            "battery_sensor_health",
            "status_sensor_health",
            "overall_sensor_health",
            "maintenance_priority",
            "maintenance_reason",
            "recommended_action",
        ]

        for col in output_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan

        return self.df
        # ======================================================
    # Fuel Sensor Health
    # ======================================================

    def evaluate_fuel_sensor(self):

        score = np.full(len(self.df), 100.0)

        invalid = (
            (self.df["fuel_level_l"] < self.MIN_FUEL) |
            (self.df["fuel_level_l"] > self.MAX_FUEL)
        )

        score[invalid] -= 60

        if "fuel_outlier" in self.df.columns:
            score[self.df["fuel_outlier"]] -= 20

        if "fuel_imputed" in self.df.columns:
            score[self.df["fuel_imputed"]] -= 10

        self.df["fuel_sensor_health"] = np.clip(score, 0, 100)

    # ======================================================
    # Current Sensor Health
    # ======================================================

    def evaluate_current_sensor(self):

        score = np.full(len(self.df), 100.0)

        invalid = (
            (self.df["current"] < self.MIN_CURRENT) |
            (self.df["current"] > self.MAX_CURRENT)
        )

        score[invalid] -= 60

        if "current_outlier" in self.df.columns:
            score[self.df["current_outlier"]] -= 20

        if "current_imputed" in self.df.columns:
            score[self.df["current_imputed"]] -= 10

        self.df["current_sensor_health"] = np.clip(score, 0, 100)

    # ======================================================
    # Battery Sensor Health
    # ======================================================

    def evaluate_battery_sensor(self):

        score = np.full(len(self.df), 100.0)

        invalid = (
            (self.df["battery_voltage"] < self.MIN_BATTERY) |
            (self.df["battery_voltage"] > self.MAX_BATTERY)
        )

        score[invalid] -= 60

        if "battery_imputed" in self.df.columns:
            score[self.df["battery_imputed"]] -= 10

        if "battery_current_mismatch" in self.df.columns:
            score[self.df["battery_current_mismatch"]] -= 15

        self.df["battery_sensor_health"] = np.clip(score, 0, 100)

    # ======================================================
    # Status Sensor Health
    # ======================================================

    def evaluate_status_sensor(self):

        score = np.full(len(self.df), 100.0)

        if "status_conflict" in self.df.columns:
            score[self.df["status_conflict"]] -= 30

        if "status_sensor_mismatch" in self.df.columns:
            score[self.df["status_sensor_mismatch"]] -= 25

        if "status_imputed" in self.df.columns:
            score[self.df["status_imputed"]] -= 15

        self.df["status_sensor_health"] = np.clip(score, 0, 100)
            # ======================================================
    # Overall Sensor Health
    # ======================================================

    def calculate_overall_health(self):
        """
        Calculate overall sensor health score.
        """

        self.df["overall_sensor_health"] = (
            self.df[
                [
                    "fuel_sensor_health",
                    "current_sensor_health",
                    "battery_sensor_health",
                    "status_sensor_health",
                ]
            ]
            .mean(axis=1)
            .round(2)
        )

    # ======================================================
    # Maintenance Recommendation
    # ======================================================

    def generate_maintenance_recommendation(self):

        priority = []
        reason = []
        action = []

        for _, row in self.df.iterrows():

            health = row["overall_sensor_health"]

            if health >= 90:
                priority.append("None")
                reason.append("Sensors operating normally")
                action.append("No action required")

            elif health >= 75:
                priority.append("Low")
                reason.append("Minor sensor degradation")
                action.append("Monitor during next inspection")

            elif health >= 50:
                priority.append("Medium")
                reason.append("Sensor quality degrading")
                action.append("Inspect affected sensor")

            elif health >= 25:
                priority.append("High")
                reason.append("Multiple sensor issues detected")
                action.append("Maintenance recommended")

            else:
                priority.append("Critical")
                reason.append("Sensor reliability very poor")
                action.append("Immediate maintenance required")

        self.df["maintenance_priority"] = priority
        self.df["maintenance_reason"] = reason
        self.df["recommended_action"] = action
            # ======================================================
    # Report
    # ======================================================

    def print_report(self):

        print("\n" + "=" * 70)
        print("SENSOR HEALTH REPORT")
        print("=" * 70)

        print("\nRows Processed:")
        print(len(self.df))

        print("\nOverall Sensor Health")

        print(
            self.df["overall_sensor_health"]
            .describe()
            .round(2)
        )

        print("\nMaintenance Priority")

        print(
            self.df["maintenance_priority"]
            .value_counts()
        )

        print("\nAverage Health:")

        print(
            round(
                self.df["overall_sensor_health"].mean(),
                2
            ),
            "%"
        )

    # ======================================================
    # Run
    # ======================================================

    def run(self):

        print("Preparing data...")
        self.prepare()

        print("Evaluating fuel sensor...")
        self.evaluate_fuel_sensor()

        print("Evaluating current sensor...")
        self.evaluate_current_sensor()

        print("Evaluating battery sensor...")
        self.evaluate_battery_sensor()

        print("Evaluating status sensor...")
        self.evaluate_status_sensor()

        print("Calculating overall health...")
        self.calculate_overall_health()

        print("Generating recommendations...")
        self.generate_maintenance_recommendation()

        self.print_report()

        return self.df
      
