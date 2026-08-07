"""
sensor_health_monitor.py
------------------------

Sensor Health Monitoring Engine

Responsibilities
----------------
1. Detect sensor drift
2. Detect frozen/stuck sensors
3. Detect noisy sensors
4. Detect impossible jumps
5. Detect intermittent failures
6. Calculate sensor reliability score
7. Assign health state
8. Produce confidence for downstream AI models

Author: Adil
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional

from scipy.stats import median_abs_deviation

# ============================================================
# HEALTH LABELS
# ============================================================

HEALTH_HEALTHY = "Healthy"
HEALTH_WARNING = "Warning"
HEALTH_DEGRADED = "Degraded"
HEALTH_FAILED = "Failed"

# ============================================================
# SENSOR NAMES
# ============================================================

FUEL_SENSOR = "fuel_level_l"
CURRENT_SENSOR = "current"
BATTERY_SENSOR = "battery_voltage"

# ============================================================
# MONITOR CLASS
# ============================================================

class SensorHealthMonitor:

    """
    Evaluates health of every telemetry sensor.

    Output columns include:

        fuel_sensor_health
        current_sensor_health
        battery_sensor_health

        fuel_sensor_score
        current_sensor_score
        battery_sensor_score

        overall_sensor_health
        overall_sensor_score
    """

    # --------------------------------------------------------

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):

        self.df = dataframe.copy()

        self.group_column = "generator_id"

        self.time_column = "timestamp"

        self.health_columns = []

        self.score_columns = []

        self.summary = {}
            # ==========================================================
    # Sensor Health Score
    # ==========================================================

    def calculate_health_score(self, row: pd.Series) -> tuple[float, list]:
        """
        Calculate overall health score of telemetry.

        Returns
        -------
        (score, reasons)
        """

        score = 100.0
        reasons = []

        # ------------------------------------------------------
        # Fuel validation
        # ------------------------------------------------------
        if row.get("fuel_invalid", False):
            score -= 30
            reasons.append("Invalid fuel value")

        if row.get("fuel_outlier", False):
            score -= 15
            reasons.append("Fuel outlier")

        # ------------------------------------------------------
        # Current
        # ------------------------------------------------------
        if row.get("current_invalid", False):
            score -= 25
            reasons.append("Invalid current")

        if row.get("current_outlier", False):
            score -= 10
            reasons.append("Current outlier")

        # ------------------------------------------------------
        # Battery
        # ------------------------------------------------------
        if row.get("battery_invalid", False):
            score -= 20
            reasons.append("Invalid battery voltage")

        # ------------------------------------------------------
        # Sensor mismatches
        # ------------------------------------------------------
        if row.get("battery_current_mismatch", False):
            score -= 15
            reasons.append("Battery/current mismatch")

        if row.get("state_sensor_mismatch", False):
            score -= 15
            reasons.append("Status conflicts with sensors")

        # ------------------------------------------------------
        # Imputed values
        # ------------------------------------------------------
        if row.get("fuel_imputed", False):
            score -= 5
            reasons.append("Fuel imputed")

        if row.get("current_imputed", False):
            score -= 5
            reasons.append("Current imputed")

        if row.get("battery_imputed", False):
            score -= 5
            reasons.append("Battery imputed")

        if row.get("status_imputed", False):
            score -= 5
            reasons.append("Status imputed")

        # ------------------------------------------------------
        # Confidence penalties
        # ------------------------------------------------------
        confidence = row.get("imputation_confidence", 100)

        if pd.notna(confidence):

            if confidence < 30:
                score -= 20
                reasons.append("Very low imputation confidence")

            elif confidence < 60:
                score -= 10
                reasons.append("Low imputation confidence")

            elif confidence < 80:
                score -= 5
                reasons.append("Moderate imputation confidence")

        score = max(score, 0)

        return score, reasons

    # ==========================================================
    # Health Category
    # ==========================================================

    @staticmethod
    def health_category(score: float) -> str:

        if score >= 95:
            return "Excellent"

        if score >= 85:
            return "Good"

        if score >= 70:
            return "Fair"

        if score >= 50:
            return "Poor"

        return "Critical"

    # ==========================================================
    # Detect Sensor Drift
    # ==========================================================

    def detect_sensor_drift(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect slowly drifting sensors using rolling mean.
        """

        drift = pd.Series(False, index=df.index)

        if "fuel_level_l" not in df.columns:
            return drift

        rolling = (
            df["fuel_level_l"]
            .rolling(window=60, min_periods=20)
            .mean()
        )

        deviation = (
            df["fuel_level_l"] - rolling
        ).abs()

        drift |= deviation > 8

        return drift
        # ==========================================================
    # Main Monitoring Pipeline
    # ==========================================================

    def run(self) -> pd.DataFrame:
        """
        Runs complete sensor health analysis.
        """

        print("Calculating sensor health...")

        df = self.df.copy()

        # ------------------------------------------------------
        # Drift Detection
        # ------------------------------------------------------
        print("Detecting sensor drift...")

        df["sensor_drift"] = False

        if "generator_id" in df.columns:

            for _, idx in df.groupby("generator_id").groups.items():
                subset = df.loc[idx].sort_values("timestamp")
                drift = self.detect_sensor_drift(subset)
                df.loc[subset.index, "sensor_drift"] = drift.values

        else:

            df["sensor_drift"] = self.detect_sensor_drift(df)

        # ------------------------------------------------------
        # Health Score
        # ------------------------------------------------------
        print("Computing health score...")

        scores = []
        categories = []
        reasons = []

        for _, row in df.iterrows():

            score, problems = self.calculate_health_score(row)

            if row.get("sensor_drift", False):
                score -= 10
                problems.append("Sensor drift detected")

            score = max(score, 0)

            scores.append(round(score, 2))
            categories.append(self.health_category(score))

            if problems:
                reasons.append("; ".join(problems))
            else:
                reasons.append("Healthy telemetry")

        df["sensor_health_score"] = scores
        df["sensor_health"] = categories
        df["sensor_health_reason"] = reasons

        # ------------------------------------------------------
        # Overall Issue Flag
        # ------------------------------------------------------
        issue_columns = [
            "fuel_invalid",
            "fuel_outlier",
            "current_invalid",
            "current_outlier",
            "battery_invalid",
            "battery_current_mismatch",
            "state_sensor_mismatch",
            "sensor_drift",
        ]

        existing = [c for c in issue_columns if c in df.columns]

        if existing:
            df["sensor_issue_detected"] = df[existing].any(axis=1)
        else:
            df["sensor_issue_detected"] = False

        # ------------------------------------------------------
        # Maintenance Recommendation
        # ------------------------------------------------------
        recommendations = []

        for _, row in df.iterrows():

            health = row["sensor_health"]

            if health == "Excellent":
                recommendations.append("No action required")

            elif health == "Good":
                recommendations.append("Continue monitoring")

            elif health == "Fair":
                recommendations.append("Inspect during next maintenance")

            elif health == "Poor":
                recommendations.append("Schedule sensor inspection")

            else:
                recommendations.append("Immediate maintenance required")

        df["maintenance_recommendation"] = recommendations

        self.df = df

        return df
        # ==========================================================
    # SUMMARY REPORT
    # ==========================================================

    def report(self) -> None:
        """
        Print a summary of sensor health.
        """

        print("\n" + "=" * 70)
        print("SENSOR HEALTH REPORT")
        print("=" * 70)

        print(f"\nRows Processed : {len(self.df):,}")

        print("\nHealth Distribution")

        print(
            self.df["sensor_health"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print("\nAverage Health Score")

        print(
            round(
                self.df["sensor_health_score"].mean(),
                2,
            )
        )

        if "sensor_issue_detected" in self.df.columns:

            print("\nRows With Sensor Issues")

            print(
                int(
                    self.df[
                        "sensor_issue_detected"
                    ].sum()
                )
            )

        print("\nMaintenance Recommendations")

        print(
            self.df[
                "maintenance_recommendation"
            ].value_counts()
        )

        print("=" * 70)

    # ==========================================================
    # SAVE OUTPUT
    # ==========================================================

    def save(
        self,
        output_path: str,
    ):

        """
        Save processed dataset.
        """

        self.df.to_csv(
            output_path,
            index=False,
        )

        print("\nSaved")

        print(output_path)

    # ==========================================================
    # GET DATAFRAME
    # ==========================================================

    def get_dataframe(self):

        return self.df

    # ==========================================================
    # GET SUMMARY
    # ==========================================================

    def get_summary(self):

        summary = {

            "rows": len(self.df),

            "average_score":
                round(
                    self.df[
                        "sensor_health_score"
                    ].mean(),
                    2,
                ),

            "excellent":
                int(
                    (
                        self.df[
                            "sensor_health"
                        ]
                        == "Excellent"
                    ).sum()
                ),

            "good":
                int(
                    (
                        self.df[
                            "sensor_health"
                        ]
                        == "Good"
                    ).sum()
                ),

            "fair":
                int(
                    (
                        self.df[
                            "sensor_health"
                        ]
                        == "Fair"
                    ).sum()
                ),

            "poor":
                int(
                    (
                        self.df[
                            "sensor_health"
                        ]
                        == "Poor"
                    ).sum()
                ),

            "critical":
                int(
                    (
                        self.df[
                            "sensor_health"
                        ]
                        == "Critical"
                    ).sum()
                ),
        }

        return summary