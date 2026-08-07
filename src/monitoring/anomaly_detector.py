"""
Anomaly Detection Engine (MVP)

Purpose
-------
Detect abnormal telemetry behaviour using engineering rules.

Input
-----
Sensor Health Dataset

Output
------
Dataset with anomaly detection columns.

Future Upgrades
---------------
- Isolation Forest
- Autoencoder
- Prophet residual analysis
- LSTM anomaly detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AnomalyDetector:
    """
    Rule-Based Anomaly Detection Engine (MVP)
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = df.copy()

        # --------------------------------------------------
        # Thresholds
        # --------------------------------------------------

        self.FUEL_THEFT_THRESHOLD = -5.0          # litres
        self.REFUEL_THRESHOLD = 5.0               # litres

        self.CURRENT_SPIKE = 50.0                 # amps

        self.BATTERY_LOW = 44.0                   # volts
        self.BATTERY_SPIKE = 5.0                  # volts

        self.STUCK_SENSOR_WINDOW = 10

    # =====================================================
    # Preparation
    # =====================================================

    def prepare(self):
        """
        Prepare dataframe before anomaly detection.
        """

        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(
                self.df["timestamp"]
            )

        self.df = (
            self.df
            .sort_values(
                ["generator_id", "timestamp"]
            )
            .reset_index(drop=True)
        )

        anomaly_columns = [

            # Fuel
            "fuel_theft_detected",
            "refueling_detected",
            "fuel_spike_detected",
            "fuel_sensor_stuck",
            "fuel_anomaly_score",

            # Current
            "current_spike_detected",
            "current_zero_running",
            "current_anomaly_score",

            # Battery
            "battery_low",
            "battery_spike",
            "battery_unstable",
            "battery_anomaly_score",

            # Generator
            "generator_state_anomaly",
            "state_anomaly_reason",

            # Telemetry
            "telemetry_anomaly",
            "telemetry_severity",

            # Overall
            "overall_anomaly",
            "anomaly_score",
            "anomaly_severity",
            "anomaly_reason",
            "recommended_action",
        ]

        for column in anomaly_columns:

            if column not in self.df.columns:

                if (
                    "reason" in column
                    or "severity" in column
                    or "action" in column
                ):
                    self.df[column] = ""

                elif (
                    "score" in column
                ):
                    self.df[column] = 0.0

                else:
                    self.df[column] = False

        return self.df
        # =====================================================
    # Fuel Anomaly Detection
    # =====================================================

    def detect_fuel_anomalies(self):
        """
        Detect fuel-related anomalies.

        Detects:
        - Fuel theft
        - Refueling
        - Fuel spikes
        - Stuck fuel sensor
        """

        print("Detecting fuel anomalies...")

        self.df["fuel_theft_detected"] = False
        self.df["refueling_detected"] = False
        self.df["fuel_spike_detected"] = False
        self.df["fuel_sensor_stuck"] = False
        self.df["fuel_anomaly_score"] = 0.0

        grouped = self.df.groupby("generator_id")

        for generator_id, group in grouped:

            index = group.index

            fuel = group["fuel_level_l"]

            fuel_delta = group["fuel_delta"]

            # ----------------------------------------------
            # Fuel Theft
            # ----------------------------------------------

            theft = fuel_delta < self.FUEL_THEFT_THRESHOLD

            self.df.loc[
                index,
                "fuel_theft_detected",
            ] = theft.values

            # ----------------------------------------------
            # Refueling
            # ----------------------------------------------

            refuel = fuel_delta > self.REFUEL_THRESHOLD

            self.df.loc[
                index,
                "refueling_detected",
            ] = refuel.values

            # ----------------------------------------------
            # Fuel Spike
            # ----------------------------------------------

            spike = fuel_delta.abs() > (
                self.REFUEL_THRESHOLD * 2
            )

            self.df.loc[
                index,
                "fuel_spike_detected",
            ] = spike.values

            # ----------------------------------------------
            # Stuck Fuel Sensor
            # ----------------------------------------------

            stuck = (
                fuel
                .rolling(
                    self.STUCK_SENSOR_WINDOW,
                    min_periods=self.STUCK_SENSOR_WINDOW,
                )
                .std()
                .fillna(1)
                == 0
            )

            self.df.loc[
                index,
                "fuel_sensor_stuck",
            ] = stuck.values

        # --------------------------------------------------
        # Fuel Anomaly Score
        # --------------------------------------------------

        score = np.zeros(len(self.df))

        score += (
            self.df["fuel_theft_detected"]
            .astype(int)
            * 40
        )

        score += (
            self.df["refueling_detected"]
            .astype(int)
            * 15
        )

        score += (
            self.df["fuel_spike_detected"]
            .astype(int)
            * 25
        )

        score += (
            self.df["fuel_sensor_stuck"]
            .astype(int)
            * 20
        )

        self.df["fuel_anomaly_score"] = np.clip(
            score,
            0,
            100,
        )
            # =====================================================
    # Current Anomaly Detection
    # =====================================================

    def detect_current_anomalies(self):
        """
        Detect current-related anomalies.

        Detects:
        - Current spikes
        - Generator running but zero current
        """

        print("Detecting current anomalies...")

        self.df["current_spike_detected"] = False
        self.df["current_zero_running"] = False
        self.df["current_anomaly_score"] = 0.0

        grouped = self.df.groupby("generator_id")

        for generator_id, group in grouped:

            index = group.index

            current_change = (
                group["current"]
                .diff()
                .fillna(0)
            )

            # ----------------------------------------------
            # Current Spike
            # ----------------------------------------------

            spike = (
                current_change.abs()
                > self.CURRENT_SPIKE
            )

            self.df.loc[
                index,
                "current_spike_detected",
            ] = spike.values

            # ----------------------------------------------
            # Running Generator but Zero Current
            # ----------------------------------------------

            if "estimated_status" in group.columns:

                running = (
                    group["estimated_status"]
                    .str.contains(
                        "Running",
                        case=False,
                        na=False,
                    )
                )

                zero_current = (
                    group["current"] <= 1
                )

                anomaly = (
                    running
                    & zero_current
                )

                self.df.loc[
                    index,
                    "current_zero_running",
                ] = anomaly.values

        # --------------------------------------------------
        # Current Anomaly Score
        # --------------------------------------------------

        score = np.zeros(len(self.df))

        score += (
            self.df["current_spike_detected"]
            .astype(int)
            * 50
        )

        score += (
            self.df["current_zero_running"]
            .astype(int)
            * 50
        )

        self.df["current_anomaly_score"] = np.clip(
            score,
            0,
            100,
        )
            # =====================================================
    # Battery Anomaly Detection
    # =====================================================

    def detect_battery_anomalies(self):
        """
        Detect battery-related anomalies.

        Detects:
        - Low battery voltage
        - Battery voltage spikes
        - Unstable battery voltage
        """

        print("Detecting battery anomalies...")

        self.df["battery_low"] = False
        self.df["battery_spike"] = False
        self.df["battery_unstable"] = False
        self.df["battery_anomaly_score"] = 0.0

        grouped = self.df.groupby("generator_id")

        for generator_id, group in grouped:

            index = group.index

            voltage = group["battery_voltage"]

            voltage_change = (
                voltage
                .diff()
                .fillna(0)
            )

            # ----------------------------------------------
            # Low Battery
            # ----------------------------------------------

            low = voltage < self.BATTERY_LOW

            self.df.loc[
                index,
                "battery_low",
            ] = low.values

            # ----------------------------------------------
            # Battery Spike
            # ----------------------------------------------

            spike = (
                voltage_change.abs()
                > self.BATTERY_SPIKE
            )

            self.df.loc[
                index,
                "battery_spike",
            ] = spike.values

            # ----------------------------------------------
            # Battery Unstable
            # ----------------------------------------------

            unstable = (
                voltage
                .rolling(
                    window=5,
                    min_periods=5,
                )
                .std()
                > 2
            ).fillna(False)

            self.df.loc[
                index,
                "battery_unstable",
            ] = unstable.values

        # --------------------------------------------------
        # Battery Anomaly Score
        # --------------------------------------------------

        score = np.zeros(len(self.df))

        score += (
            self.df["battery_low"]
            .astype(int)
            * 40
        )

        score += (
            self.df["battery_spike"]
            .astype(int)
            * 30
        )

        score += (
            self.df["battery_unstable"]
            .astype(int)
            * 30
        )

        self.df["battery_anomaly_score"] = np.clip(
            score,
            0,
            100,
        )
            # =====================================================
    # Generator State Anomaly Detection
    # =====================================================

    def detect_generator_state_anomalies(self):
        """
        Detect generator operating state anomalies.
        """

        print("Detecting generator state anomalies...")

        self.df["generator_state_anomaly"] = False
        self.df["state_anomaly_reason"] = ""

        for idx, row in self.df.iterrows():

            anomaly = False
            reasons = []

            estimated = str(
                row.get(
                    "estimated_status",
                    "",
                )
            ).lower()

            current = row.get(
                "current",
                np.nan,
            )

            fuel_delta = row.get(
                "fuel_delta",
                0,
            )

            battery = row.get(
                "battery_voltage",
                np.nan,
            )

            # ------------------------------------------
            # Running but zero current
            # ------------------------------------------

            if (
                "running" in estimated
                and current <= 1
            ):

                anomaly = True

                reasons.append(
                    "Running but zero current"
                )

            # ------------------------------------------
            # Stopped but drawing current
            # ------------------------------------------

            if (
                "stopped" in estimated
                and current > 10
            ):

                anomaly = True

                reasons.append(
                    "Stopped but current detected"
                )

            # ------------------------------------------
            # Stopped but fuel decreasing
            # ------------------------------------------

            if (
                "stopped" in estimated
                and fuel_delta < -0.2
            ):

                anomaly = True

                reasons.append(
                    "Fuel decreasing while stopped"
                )

            # ------------------------------------------
            # Running but battery missing
            # ------------------------------------------

            if (
                "running" in estimated
                and (
                    pd.isna(battery)
                    or battery <= 0
                )
            ):

                anomaly = True

                reasons.append(
                    "Battery unavailable"
                )

            self.df.at[
                idx,
                "generator_state_anomaly",
            ] = anomaly

            self.df.at[
                idx,
                "state_anomaly_reason",
            ] = ", ".join(reasons)
                # =====================================================
    # Telemetry Anomaly Detection
    # =====================================================

    def detect_telemetry_anomalies(self):
        """
        Detect telemetry communication anomalies.

        Uses outputs produced by the Telemetry Validator.
        """

        print("Detecting telemetry anomalies...")

        self.df["telemetry_anomaly"] = False
        self.df["telemetry_severity"] = ""

        for idx, row in self.df.iterrows():

            anomaly = False
            reasons = []

            # ------------------------------------------
            # Duplicate Timestamp
            # ------------------------------------------

            if row.get("timestamp_duplicate", False):

                anomaly = True

                reasons.append("Duplicate Timestamp")

            # ------------------------------------------
            # Timestamp Gap
            # ------------------------------------------

            if row.get("timestamp_gap", False):

                anomaly = True

                reasons.append("Missing Telemetry")

            # ------------------------------------------
            # Low Telemetry Quality
            # ------------------------------------------

            if row.get(
                "telemetry_quality_score",
                100,
            ) < 80:

                anomaly = True

                reasons.append("Low Quality")

            # ------------------------------------------
            # Determine Severity
            # ------------------------------------------

            if len(reasons) == 0:

                severity = ""

            elif len(reasons) == 1:

                severity = "Low"

            elif len(reasons) == 2:

                severity = "Medium"

            else:

                severity = "High"

            self.df.at[
                idx,
                "telemetry_anomaly",
            ] = anomaly

            self.df.at[
                idx,
                "telemetry_severity",
            ] = severity
                # =====================================================
    # Overall Anomaly Assessment
    # =====================================================

    def calculate_overall_anomaly(self):
        """
        Combine all anomaly detectors into one score.
        """

        print("Calculating overall anomaly score...")

        # -------------------------------
        # Overall Score
        # -------------------------------

        self.df["anomaly_score"] = (
            self.df["fuel_anomaly_score"] * 0.35
            + self.df["current_anomaly_score"] * 0.20
            + self.df["battery_anomaly_score"] * 0.20
            + self.df["generator_state_anomaly"].astype(int) * 15
            + self.df["telemetry_anomaly"].astype(int) * 10
        ).round(2)

        self.df["anomaly_score"] = self.df[
            "anomaly_score"
        ].clip(0, 100)

        # -------------------------------
        # Overall Anomaly Flag
        # -------------------------------

        self.df["overall_anomaly"] = (
            self.df["anomaly_score"] > 20
        )

        # -------------------------------
        # Severity
        # -------------------------------

        severity = []

        for score in self.df["anomaly_score"]:

            if score == 0:
                severity.append("None")

            elif score < 25:
                severity.append("Low")

            elif score < 50:
                severity.append("Medium")

            elif score < 75:
                severity.append("High")

            else:
                severity.append("Critical")

        self.df["anomaly_severity"] = severity

    # =====================================================
    # Recommendations
    # =====================================================

    def generate_recommendations(self):
        """
        Generate anomaly reason and recommendation.
        """

        print("Generating recommendations...")

        reasons = []
        actions = []

        for _, row in self.df.iterrows():

            detected = []
            response = []

            # Fuel

            if row["fuel_theft_detected"]:
                detected.append("Fuel theft suspected")
                response.append("Inspect fuel tank")

            if row["refueling_detected"]:
                detected.append("Refueling event")
                response.append("Verify refill")

            if row["fuel_sensor_stuck"]:
                detected.append("Fuel sensor stuck")
                response.append("Inspect fuel sensor")

            # Current

            if row["current_spike_detected"]:
                detected.append("Current spike")
                response.append("Inspect electrical load")

            if row["current_zero_running"]:
                detected.append("Zero current while running")
                response.append("Inspect current sensor")

            # Battery

            if row["battery_low"]:
                detected.append("Low battery")
                response.append("Check battery")

            if row["battery_spike"]:
                detected.append("Battery voltage spike")
                response.append("Inspect charging system")

            if row["battery_unstable"]:
                detected.append("Battery unstable")
                response.append("Inspect battery")

            # Generator

            if row["generator_state_anomaly"]:
                detected.append("Generator state conflict")
                response.append("Verify generator state")

            # Telemetry

            if row["telemetry_anomaly"]:
                detected.append("Telemetry issue")
                response.append("Inspect communication")

            if len(detected) == 0:

                reasons.append("No anomaly detected")
                actions.append("No action required")

            else:

                reasons.append(
                    "; ".join(detected)
                )

                actions.append(
                    "; ".join(
                        sorted(set(response))
                    )
                )

        self.df["anomaly_reason"] = reasons
        self.df["recommended_action"] = actions

    # =====================================================
    # Report
    # =====================================================

    def print_report(self):

        print("\n" + "=" * 70)
        print("ANOMALY DETECTION REPORT")
        print("=" * 70)

        print("\nRows Processed:")
        print(len(self.df))

        print("\nOverall Anomalies:")
        print(
            self.df["overall_anomaly"]
            .value_counts()
        )

        print("\nSeverity Distribution:")
        print(
            self.df["anomaly_severity"]
            .value_counts()
        )

        print("\nAverage Anomaly Score:")
        print(
            round(
                self.df["anomaly_score"].mean(),
                2,
            )
        )

    # =====================================================
    # Run
    # =====================================================

    def run(self):

        self.prepare()

        self.detect_fuel_anomalies()

        self.detect_current_anomalies()

        self.detect_battery_anomalies()

        self.detect_generator_state_anomalies()

        self.detect_telemetry_anomalies()

        self.calculate_overall_anomaly()

        self.generate_recommendations()

        self.print_report()

        return self.df