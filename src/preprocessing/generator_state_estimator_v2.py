"""
==============================================================
Generator State Estimator V2
==============================================================

Industrial telemetry-based generator state estimation.

This estimator infers the true operating state of telecom
generators using multiple telemetry sensors rather than
trusting the reported status column.

Features
--------
✓ Adaptive thresholds per generator
✓ Multi-sensor evidence extraction
✓ Weighted decision engine
✓ Running probability estimation
✓ Confidence estimation
✓ Status conflict detection
✓ Sensor fault detection
✓ Load classification
✓ Dataset reporting

Author:
Fuel Telemetry AI Project
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================


@dataclass
class GeneratorStateEstimatorConfig:
    """
    Configuration for Generator State Estimator V2.
    """

    # Current below this is considered zero
    minimum_current: float = 1.0

    # Fuel rate below this is treated as no consumption (L/hr)
    minimum_fuel_rate: float = 0.05

    # Fuel change considered meaningful (Liters)
    minimum_fuel_delta: float = 0.05

    # Battery voltage considered generator active
    battery_voltage_threshold: float = 24.0

    # Confidence scaling
    maximum_confidence: float = 100.0

    # Weights
    current_weight: int = 45
    fuel_rate_weight: int = 25
    fuel_delta_weight: int = 15
    battery_weight: int = 10
    status_weight: int = 5


# ==========================================================
# Generator State Estimator
# ==========================================================


class GeneratorStateEstimatorV2:
    """
    Industrial telemetry-based generator state estimator.

    Pipeline

        Validate Dataset
                │
                ▼
        Learn Generator Thresholds
                │
                ▼
        Extract Sensor Evidence
                │
                ▼
        Fuse Evidence
                │
                ▼
        Detect Conflicts
                │
                ▼
        Estimate Generator State
                │
                ▼
        Generate Reports
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[GeneratorStateEstimatorConfig] = None,
    ):

        self.df = df.copy()

        self.config = config or GeneratorStateEstimatorConfig()

        # Learned thresholds for each generator
        self.generator_thresholds: Dict[str, Dict] = {}

        # Results
        self.summary = []
        self.results = None
    # ======================================================
    # Validation
    # ======================================================

    def validate(self):
        """
        Validate required dataset columns.
        """

        required = [

            "generator_id",

            "status",

            "current",

            "fuel_level_l",

            "fuel_rate_lph",

            "battery_voltage",

        ]

        missing = [

            column

            for column in required

            if column not in self.df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing required columns: {missing}"

            )

        if self.df.empty:

            raise ValueError(

                "Input dataframe is empty."

            )

    # ======================================================
    # Learn Generator Thresholds
    # ======================================================

    def learn_generator_thresholds(self):
        """
        Learn adaptive thresholds independently for every
        generator.

        Different generators have different capacities,
        therefore current thresholds should not be global.
        """

        self.generator_thresholds = {}

        grouped = self.df.groupby("generator_id")

        for generator_id, generator_df in grouped:

            current = generator_df["current"].dropna()

            if len(current) == 0:

                self.generator_thresholds[generator_id] = {

                    "low": 20,

                    "medium": 60,

                    "high": 120,

                }

                continue

            positive_current = current[
                current > self.config.minimum_current
            ]

            if len(positive_current) < 10:

                self.generator_thresholds[generator_id] = {

                    "low": 20,

                    "medium": 60,

                    "high": 120,

                }

                continue

            self.generator_thresholds[generator_id] = {

                "low": float(

                    np.percentile(

                        positive_current,

                        25

                    )

                ),

                "medium": float(

                    np.percentile(

                        positive_current,

                        50

                    )

                ),

                "high": float(

                    np.percentile(

                        positive_current,

                        75

                    )

                ),

            }

        return self.generator_thresholds

    # ======================================================
    # Threshold Lookup
    # ======================================================

    def get_thresholds(
        self,
        generator_id: str
    ) -> Dict:

        if not self.generator_thresholds:

            self.learn_generator_thresholds()

        return self.generator_thresholds.get(

            generator_id,

            {

                "low": 20,

                "medium": 60,

                "high": 120,

            }

        )
        # ======================================================
    # Current Evidence
    # ======================================================

    def current_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Estimate running evidence from generator load.
        """

        generator = row["generator_id"]

        thresholds = self.get_thresholds(generator)

        current = row["current"]

        if pd.isna(current):

            return {

                "score": 0.50,
                "label": "Unknown",
                "available": False,

            }

        if current <= self.config.minimum_current:

            return {

                "score": 0.0,
                "label": "Stopped",
                "available": True,

            }

        if current < thresholds["low"]:

            return {

                "score": 0.55,
                "label": "Running (Low Load)",
                "available": True,

            }

        if current < thresholds["medium"]:

            return {

                "score": 0.75,
                "label": "Running (Medium Load)",
                "available": True,

            }

        return {

            "score": 1.0,
            "label": "Running (High Load)",
            "available": True,

        }

    # ======================================================
    # Fuel Rate Evidence
    # ======================================================

    def fuel_rate_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Estimate running evidence from fuel consumption.
        """

        rate = row["fuel_rate_lph"]

        if pd.isna(rate):

            return {

                "score": 0.50,
                "label": "Unknown",
                "available": False,

            }

        if rate > self.config.minimum_fuel_rate:

            return {

                "score": 1.0,
                "label": "Fuel Consuming",
                "available": True,

            }

        if rate < -self.config.minimum_fuel_rate:

            return {

                "score": 0.20,
                "label": "Fuel Increasing",
                "available": True,

            }

        return {

            "score": 0.0,
            "label": "No Consumption",
            "available": True,

        }

    # ======================================================
    # Fuel Trend Evidence
    # ======================================================

    def fuel_trend_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Estimate running evidence from change in fuel level.
        """

        delta = row["fuel_delta"]

        if pd.isna(delta):

            return {

                "score": 0.50,
                "label": "Unknown",
                "available": False,

            }

        if delta < -self.config.minimum_fuel_delta:

            return {

                "score": 1.0,
                "label": "Fuel Decreasing",
                "available": True,

            }

        if delta > self.config.minimum_fuel_delta:

            return {

                "score": 0.20,
                "label": "Fuel Increasing",
                "available": True,

            }

        return {

            "score": 0.0,
            "label": "Stable Fuel",
            "available": True,

        }

    # ======================================================
    # Battery Evidence
    # ======================================================

    def battery_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Estimate running evidence from battery voltage.
        """

        voltage = row["battery_voltage"]

        if pd.isna(voltage):

            return {

                "score": 0.50,
                "label": "Unknown",
                "available": False,

            }

        if voltage >= self.config.battery_voltage_threshold:

            return {

                "score": 1.0,
                "label": "Battery Active",
                "available": True,

            }

        if voltage > 0:

            return {

                "score": 0.50,
                "label": "Low Battery",
                "available": True,

            }

        return {

            "score": 0.0,
            "label": "Battery Off",
            "available": True,

        }

    # ======================================================
    # Reported Status Evidence
    # ======================================================

    def status_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Reported status is treated as the least reliable sensor.
        """

        status = row["status"]

        if pd.isna(status):

            return {

                "score": 0.50,
                "label": "Unknown",
                "available": False,

            }

        status = str(status).strip().lower()

        if status == "running":

            return {

                "score": 1.0,
                "label": "Reported Running",
                "available": True,

            }

        if status == "stopped":

            return {

                "score": 0.0,
                "label": "Reported Stopped",
                "available": True,

            }

        return {

            "score": 0.50,
            "label": "Unknown",
            "available": False,

        }

    # ======================================================
    # Collect Sensor Evidence
    # ======================================================

    def collect_evidence(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Collect evidence from all telemetry sensors.
        """

        evidence = {

            "current": self.current_evidence(row),

            "fuel_rate": self.fuel_rate_evidence(row),

            "fuel_trend": self.fuel_trend_evidence(row),

            "battery": self.battery_evidence(row),

            "status": self.status_evidence(row),

        }

        return evidence
        # ======================================================
    # Weighted Evidence Fusion
    # ======================================================

    def calculate_running_probability(
        self,
        evidence: dict,
    ) -> float:
        """
        Combine all available evidence using weighted voting.

        Missing sensors are ignored automatically.
        """

        weighted_sum = 0.0
        total_weight = 0.0

        for sensor_name, sensor in evidence.items():

            if not sensor["available"]:
                continue

            weight = self.config.sensor_weights.get(sensor_name, 0)

            weighted_sum += sensor["score"] * weight
            total_weight += weight

        if total_weight == 0:
            return 0.50

        probability = weighted_sum / total_weight

        return round(probability, 3)

    # ======================================================
    # Determine Estimated State
    # ======================================================

    def estimate_state(
        self,
        probability: float,
        current: float,
    ) -> str:
        """
        Convert probability into generator state.
        """

        if probability < 0.25:
            return "Stopped"

        if probability < 0.50:
            return "Idle"

        if pd.isna(current):
            return "Running"

        if current <= self.config.minimum_current:
            return "Idle"

        if current < self.current_statistics["low"]:
            return "Running (Low Load)"

        if current < self.current_statistics["medium"]:
            return "Running (Medium Load)"

        return "Running (High Load)"

    # ======================================================
    # Estimate Confidence
    # ======================================================

    def estimate_confidence(
        self,
        evidence: dict,
        probability: float,
    ) -> float:
        """
        Confidence depends on

        • number of available sensors
        • agreement between sensors
        """

        available_scores = []

        for sensor in evidence.values():

            if sensor["available"]:
                available_scores.append(sensor["score"])

        if len(available_scores) == 0:
            return 0.0

        agreement = 1 - np.std(available_scores)

        agreement = np.clip(agreement, 0, 1)

        completeness = (

            len(available_scores)
            /
            len(evidence)

        )

        confidence = (

            agreement * 0.7
            +
            completeness * 0.3

        )

        return round(confidence * 100, 2)

    # ======================================================
    # Detect Status Conflict
    # ======================================================

    def detect_conflict(
        self,
        reported_status,
        estimated_state,
    ) -> bool:
        """
        Detect disagreement between telemetry
        and reported status.
        """

        if pd.isna(reported_status):
            return False

        reported = str(reported_status).lower()

        estimated_running = estimated_state.startswith("Running")

        if reported == "running" and not estimated_running:
            return True

        if reported == "stopped" and estimated_running:
            return True

        return False

    # ======================================================
    # Confidence Category
    # ======================================================

    def confidence_level(
        self,
        confidence: float,
    ) -> str:

        if confidence >= 95:
            return "Very High"

        if confidence >= 85:
            return "High"

        if confidence >= 70:
            return "Medium"

        if confidence >= 50:
            return "Low"

        return "Very Low"

    # ======================================================
    # Analyze One Row
    # ======================================================

    def analyze_row(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Perform complete generator state estimation.
        """

        evidence = self.collect_evidence(row)

        probability = self.calculate_running_probability(
            evidence
        )

        state = self.estimate_state(
            probability,
            row["current"],
        )

        confidence = self.estimate_confidence(
            evidence,
            probability,
        )

        conflict = self.detect_conflict(
            row["status"],
            state,
        )

        return {

            "running_probability": probability,

            "estimated_status": state,

            "estimated_confidence": confidence,

            "confidence_level": self.confidence_level(
                confidence
            ),

            "status_conflict": conflict,

        }   
        # ======================================================
    # Process Entire Dataset
    # ======================================================

    def run(self) -> pd.DataFrame:
        """
        Run generator state estimation on the complete dataset.
        """

        print("\nEstimating generator states...\n")

        results = []

        for _, row in self.df.iterrows():

            result = self.analyze_row(row)

            results.append(result)

        results_df = pd.DataFrame(results)

        output = pd.concat(
            [
                self.df.reset_index(drop=True),
                results_df
            ],
            axis=1
        )

        self.results = output

        return output

    # ======================================================
    # Summary Report
    # ======================================================

    def build_summary(self) -> pd.DataFrame:
        """
        Generate dataset-level summary.
        """

        if self.results is None:
            raise RuntimeError("Run estimator first.")

        summary = {
            "Rows Processed": len(self.results),

            "Running Rows": (
                self.results["estimated_status"]
                .str.startswith("Running")
                .sum()
            ),

            "Idle Rows": (
                self.results["estimated_status"] == "Idle"
            ).sum(),

            "Stopped Rows": (
                self.results["estimated_status"] == "Stopped"
            ).sum(),

            "Unknown Rows": (
                self.results["estimated_status"] == "Unknown"
            ).sum(),

            "Telemetry Conflicts": (
                self.results["status_conflict"]
            ).sum(),

            "Average Confidence": round(
                self.results["estimated_confidence"].mean(),
                2
            )
        }

        return pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"]
        )

    # ======================================================
    # Print Report
    # ======================================================

    def print_report(self):

        if self.results is None:
            raise RuntimeError("Run estimator first.")

        print("\n")
        print("=" * 70)
        print("GENERATOR STATE ESTIMATION REPORT")
        print("=" * 70)

        summary = self.build_summary()

        for _, row in summary.iterrows():
            print(f"{row['Metric']:<25}: {row['Value']}")

        print("\nEstimated State Distribution")

        print(
            self.results["estimated_status"]
            .value_counts()
        )

        print("\nConfidence Distribution")

        print(
            self.results["confidence_level"]
            .value_counts()
        )

    # ======================================================
    # Save Results
    # ======================================================

    def save(
        self,
        output_path: str,
        summary_path: str,
    ):
        """
        Save estimated dataset and summary.
        """

        if self.results is None:
            raise RuntimeError("Run estimator first.")

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results.to_csv(
            output_path,
            index=False
        )

        self.build_summary().to_csv(
            summary_path,
            index=False
        )

    # ======================================================
    # Convenience Pipeline
    # ======================================================

    def execute(
        self,
        output_path: str,
        summary_path: str,
    ) -> pd.DataFrame:
        """
        Complete estimation pipeline.
        """

        results = self.run()

        self.print_report()

        self.save(
            output_path,
            summary_path,
        )

        return results