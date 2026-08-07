"""
==============================================================
Telemetry Quality Scorer
==============================================================

Evaluates telemetry reliability for each generator.

Metrics:
---------
✔ Missing packet rate
✔ Reporting interval stability
✔ Sensor availability
✔ Communication health
✔ Overall telemetry quality score

Output:
--------
generator_id
quality_score
communication_health
issues

Fuel Telemetry AI Project
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class TelemetryQualityScorer:


    def __init__(
        self,
        df: pd.DataFrame,
        interval_dictionary: Dict[str, int]
    ):

        self.df = df.copy()

        self.interval_dictionary = interval_dictionary

        self.results = []


    # ======================================================
    # Validation
    # ======================================================

    def _validate(self):

        required = [
            "generator_id",
            "timestamp"
        ]

        missing = [
            c for c in required
            if c not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )


    # ======================================================
    # Packet Analysis
    # ======================================================

    def _packet_quality(
        self,
        generator_df,
        generator_id
    ):

        expected_interval = (
            self.interval_dictionary
            .get(
                generator_id,
                60
            )
        )


        timestamps = (
            generator_df
            .sort_values("timestamp")
            ["timestamp"]
        )


        diffs = (
            timestamps
            .diff()
            .dt.total_seconds()
            .dropna()
        )


        if len(diffs)==0:

            return 100


        expected_packets = (
            diffs.sum()
            /
            expected_interval
        )


        actual_packets = len(diffs)


        missing_ratio = (
            max(
                0,
                expected_packets - actual_packets
            )
            /
            expected_packets
        )


        score = (
            100 -
            missing_ratio * 100
        )


        return max(
            0,
            min(
                100,
                score
            )
        )


    # ======================================================
    # Interval Stability
    # ======================================================

    def _interval_quality(
        self,
        generator_df,
        generator_id
    ):


        expected = (
            self.interval_dictionary
            .get(
                generator_id,
                60
            )
        )


        timestamps = (
            generator_df
            .sort_values("timestamp")
            ["timestamp"]
        )


        diffs = (
            timestamps
            .diff()
            .dt.total_seconds()
            .dropna()
        )


        if len(diffs)==0:

            return 0


        deviation = (
            abs(
                diffs.mean()
                -
                expected
            )
            /
            expected
        )


        score = (
            100 -
            deviation * 100
        )


        return max(
            0,
            min(
                100,
                score
            )
        )


    # ======================================================
    # Sensor Availability
    # ======================================================

    def _sensor_quality(
        self,
        generator_df
    ):


        sensor_columns = [

            "fuel_level_l",

            "current",

            "battery_voltage"

        ]


        available = [

            c
            for c in sensor_columns
            if c in generator_df.columns

        ]


        if not available:

            return 50


        missing_ratio = (

            generator_df[available]
            .isna()
            .mean()
            .mean()

        )


        return max(
            0,
            100 -
            missing_ratio * 100
        )


    # ======================================================
    # Health Classification
    # ======================================================

    def _health(
        self,
        score
    ):

        if score >= 90:
            return "Excellent"

        if score >= 75:
            return "Good"

        if score >= 50:
            return "Warning"

        return "Critical"



    # ======================================================
    # Issue Detection
    # ======================================================

    def _issues(
        self,
        packet,
        interval,
        sensor
    ):

        issues=[]


        if packet < 90:
            issues.append(
                "missing_packets"
            )


        if interval < 80:
            issues.append(
                "unstable_reporting"
            )


        if sensor < 90:
            issues.append(
                "sensor_missing_values"
            )


        if not issues:

            issues.append(
                "healthy"
            )


        return issues



    # ======================================================
    # Analyze Generator
    # ======================================================

    def _analyze_generator(
        self,
        generator_df
    ):

        generator_id = (
            generator_df
            ["generator_id"]
            .iloc[0]
        )


        packet_score = self._packet_quality(
            generator_df,
            generator_id
        )


        interval_score = self._interval_quality(
            generator_df,
            generator_id
        )


        sensor_score = self._sensor_quality(
            generator_df
        )


        final_score = round(

            (
                packet_score * 0.4
                +
                interval_score * 0.4
                +
                sensor_score * 0.2

            ),

            2

        )


        return {

            "generator_id":
                generator_id,

            "packet_score":
                round(packet_score,2),

            "interval_score":
                round(interval_score,2),

            "sensor_score":
                round(sensor_score,2),

            "quality_score":
                final_score,

            "communication_health":
                self._health(final_score),

            "issues":
                self._issues(
                    packet_score,
                    interval_score,
                    sensor_score
                )

        }



    # ======================================================
    # Run
    # ======================================================

    def run(self):

        self._validate()

        self.results=[]


        for _, group in self.df.groupby(
            "generator_id"
        ):

            self.results.append(

                self._analyze_generator(
                    group
                )

            )


        return pd.DataFrame(
            self.results
        )


    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        filepath
    ):

        result = self.run()

        result.to_csv(
            filepath,
            index=False
        )