"""
==============================================================
Reporting Interval Detector
==============================================================

Detects the true telemetry reporting interval for every generator.

Features
--------
✔ Automatic interval discovery
✔ Robust against missing packets
✔ Robust against duplicate packets
✔ Robust against communication outages
✔ Histogram based clustering
✔ Confidence estimation
✔ Generator diagnostics
✔ Exportable summary

Author:
Fuel Telemetry AI Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================


@dataclass
class ReportingIntervalConfig:

    # Ignore intervals larger than this (communication outage)
    max_valid_interval: int = 300

    # Ignore intervals smaller than this
    min_valid_interval: float = 1.0

    # Histogram bin width (seconds)
    histogram_bin_size: int = 1

    # Radius around dominant interval
    cluster_radius: int = 3

    # Minimum observations required
    minimum_samples: int = 30

    # Default interval if estimation fails
    default_interval: int = 60


# ==========================================================
# Detector
# ==========================================================


class ReportingIntervalDetector:

    def __init__(

        self,

        df: pd.DataFrame,

        config: Optional[ReportingIntervalConfig] = None

    ):

        self.df = df.copy()

        self.config = config or ReportingIntervalConfig()

        self.summary = []

        self.interval_dictionary = {}

    # ======================================================
    # Validation
    # ======================================================

    def _validate(self):

        required = [

            "generator_id",

            "timestamp"

        ]

        missing = [

            c

            for c in required

            if c not in self.df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns : {missing}"

            )

    # ======================================================
    # Timestamp Differences
    # ======================================================

    def _calculate_differences(

        self,

        generator_df: pd.DataFrame

    ) -> np.ndarray:

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

        diffs = diffs[

            diffs > self.config.min_valid_interval

        ]

        diffs = diffs[

            diffs <= self.config.max_valid_interval

        ]

        return diffs.to_numpy()

    # ======================================================
    # Histogram
    # ======================================================

    def _build_histogram(

        self,

        diffs: np.ndarray

    ):

        if len(diffs) == 0:

            return None, None

        max_value = int(np.ceil(diffs.max()))

        bins = np.arange(

            0,

            max_value +

            self.config.histogram_bin_size,

            self.config.histogram_bin_size

        )

        hist, edges = np.histogram(

            diffs,

            bins=bins

        )

        return hist, edges

    # ======================================================
    # Dominant Peak
    # ======================================================

    def _dominant_peak(

        self,

        hist,

        edges

    ):

        if hist is None:

            return None

        peak_index = np.argmax(hist)

        lower = edges[peak_index]

        upper = edges[peak_index + 1]

        return (

            lower +

            upper

        ) / 2

    # ======================================================
    # Cluster Samples
    # ======================================================

    def _cluster_samples(

        self,

        diffs: np.ndarray,

        center: float

    ):

        radius = self.config.cluster_radius

        return diffs[

            np.abs(

                diffs - center

            ) <= radius

        ]

    # ======================================================
    # Confidence
    # ======================================================

    def _confidence(

        self,

        cluster,

        total

    ):

        if total == 0:

            return 0

        return round(

            len(cluster)

            / total

            * 100,

            2

        )

    # ======================================================
    # Base Reporting Interval Detection
    # ======================================================

    def _base_interval_detection(
        self,
        diffs: np.ndarray
    ):

        if len(diffs) == 0:
            return None, 0

        # Round seconds
        rounded = np.round(diffs).astype(int)

        # Remove extreme communication gaps
        rounded = rounded[
            rounded <= self.config.max_valid_interval
        ]

        if len(rounded) == 0:
            return None, 0

        # Frequency distribution
        values, counts = np.unique(
            rounded,
            return_counts=True
        )

        # Highest frequency interval
        peak_index = np.argmax(counts)

        detected_interval = values[
            peak_index
        ]


        # Cluster around detected interval
        # Calculate jitter tolerant confidence

        tolerance = max(
            5,
            detected_interval * 0.08
        )

        cluster = rounded[
            np.abs(
                rounded - detected_interval
            )
            <= tolerance
        ]

        cluster_ratio = (
            len(cluster)
            /
            len(rounded)
        ) * 100

        jitter_penalty = (
            np.std(cluster)
            /
            detected_interval
        ) * 100

        confidence = (
            cluster_ratio
            -
            jitter_penalty
        )

        confidence = max(
            0,
            min(
                confidence,
                100
            )
        )

        return (
            int(detected_interval),
            round(confidence, 2)
        )

    # ======================================================
    # Estimate Interval
    # ======================================================

    def _estimate_interval(
        self,
        diffs: np.ndarray
    ):

        if len(diffs) < self.config.minimum_samples:

            return {

                "interval":
                    self.config.default_interval,

                "confidence":
                    0,

                "samples":
                    len(diffs),

                "cluster":
                    np.array([])

            }

        base_interval, base_confidence = (

            self._base_interval_detection(
                diffs
            )

        )

        if base_interval is None:

            base_interval = (
                self.config.default_interval
            )

        # create cluster around detected base interval

        tolerance = max(
            10,
            base_interval * 0.25
        )

        cluster = diffs[

            np.abs(
                diffs - base_interval
            )

            <= tolerance

        ]

        refined_interval = int(

            round(

                np.median(
                    cluster
                )

            )

        ) if len(cluster) else base_interval

        return {

            "interval":

                refined_interval,


            "confidence":

                base_confidence,


            "samples":

                len(diffs),


            "cluster":

                cluster,


            "peak":

                base_interval,


            "histogram":

                None,


            "edges":

                None

        }
        # ======================================================
    # Stability Classification
    # ======================================================

    def _stability(self, confidence: float) -> str:

        if confidence >= 95:
            return "Excellent"

        if confidence >= 85:
            return "Good"

        if confidence >= 70:
            return "Fair"

        if confidence >= 50:
            return "Poor"

        return "Unstable"

    # ======================================================
    # Jitter Statistics
    # ======================================================

    def _jitter_statistics(self, cluster: np.ndarray) -> dict:

        if len(cluster) == 0:

            return {

                "minimum": np.nan,
                "maximum": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan

            }

        return {

            "minimum": round(cluster.min(), 2),

            "maximum": round(cluster.max(), 2),

            "mean": round(cluster.mean(), 2),

            "median": round(np.median(cluster), 2),

            "std": round(cluster.std(), 2)

        }

    # ======================================================
    # Analyze Generator
    # ======================================================

    def analyze_generator(
        self,
        generator_df: pd.DataFrame
    ) -> dict:

        generator = generator_df["generator_id"].iloc[0]

        diffs = self._calculate_differences(
            generator_df
        )

        result = self._estimate_interval(
            diffs
        )

        stats = self._jitter_statistics(
            result["cluster"]
        )

        report = {

            "Generator":

                generator,

            "Reporting Interval (sec)":

                result["interval"],

            "Confidence (%)":

                result["confidence"],

            "Stability":

                self._stability(
                    result["confidence"]
                ),

            "Samples Used":

                result["samples"],

            "Cluster Samples":

                len(result["cluster"]),

            "Minimum Interval":

                stats["minimum"],

            "Median Interval":

                stats["median"],

            "Mean Interval":

                stats["mean"],

            "Maximum Interval":

                stats["maximum"],

            "Std Deviation":

                stats["std"],

            "Total Records":

                len(generator_df)

        }

        self.interval_dictionary[
            generator
        ] = result["interval"]

        return report

    # ======================================================
    # Analyze All Generators
    # ======================================================

    def analyze(self):

        self.summary = []

        grouped = self.df.groupby(
            "generator_id",
            sort=True
        )

        for _, generator_df in grouped:

            report = self.analyze_generator(
                generator_df
            )

            self.summary.append(
                report
            )

        return pd.DataFrame(
            self.summary
        )

    # ======================================================
    # Interval Class Normalization
    # ======================================================

    def _interval_classification(
        self,
        summary: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Groups similar reporting intervals into
        common telemetry classes.

        Example:

        71,74,76,75 -> 75 sec class
        15,15,16 -> 15 sec class
        """

        intervals = summary[
            "Reporting Interval (sec)"
        ].values

        normalized = []

        used = set()

        for interval in intervals:

            if interval in used:
                continue

            # group intervals within tolerance

            cluster = summary[

                abs(

                    summary[
                        "Reporting Interval (sec)"
                    ]

                    - interval

                )

                <= 5

            ]

            class_interval = int(

                round(

                    cluster[
                        "Reporting Interval (sec)"
                    ]

                    .median()

                )

            )

            for value in cluster[
                "Reporting Interval (sec)"
            ]:

                used.add(value)

            normalized.extend(

                [class_interval]

                *

                len(cluster)

            )

        # fallback safety

        if len(normalized) != len(summary):

            normalized = (

                summary[
                    "Reporting Interval (sec)"
                ]

                .tolist()

            )

        summary = summary.copy()

        summary[

            "Normalized Interval (sec)"

        ] = normalized

        return summary

    # ======================================================
    # Summary
    # ======================================================

    def get_summary(self):

        if len(self.summary) == 0:

            return self.analyze()

        return pd.DataFrame(
            self.summary
        )

    # ======================================================
    # Dictionary
    # ======================================================

    def get_interval_dictionary(self):

        if len(self.interval_dictionary) == 0:

            self.analyze()

        return self.interval_dictionary

    # ======================================================
    # Lookup
    # ======================================================

    def get_interval(
        self,
        generator_id: str
    ):

        if len(self.interval_dictionary) == 0:

            self.analyze()

        return self.interval_dictionary.get(

            generator_id,

            self.config.default_interval

        )

    # ======================================================
    # Save Summary
    # ======================================================

    def save_summary(
        self,
        filepath: str
    ):

        summary = self.get_summary()

        summary.to_csv(
            filepath,
            index=False
        )

    # ======================================================
    # Pretty Report
    # ======================================================

    def print_report(self):

        summary = self.get_summary()

        print()
        print("=" * 75)
        print("REPORTING INTERVAL DETECTION")
        print("=" * 75)

        print(summary)

        print()

        print(
            f"Generators : {len(summary)}"
        )

        print(
            f"Average Interval : "
            f"{summary['Reporting Interval (sec)'].mean():.2f} sec"
        )

        print(
            f"Average Confidence : "
            f"{summary['Confidence (%)'].mean():.2f}%"
        )

        excellent = (
            summary["Stability"] == "Excellent"
        ).sum()

        good = (
            summary["Stability"] == "Good"
        ).sum()

        fair = (
            summary["Stability"] == "Fair"
        ).sum()

        poor = (
            summary["Stability"] == "Poor"
        ).sum()

        unstable = (
            summary["Stability"] == "Unstable"
        ).sum()

        print()

        print("Stability Summary")

        print(f"Excellent : {excellent}")
        print(f"Good      : {good}")
        print(f"Fair      : {fair}")
        print(f"Poor      : {poor}")
        print(f"Unstable  : {unstable}")

    # ======================================================
    # Pipeline Entry
    # ======================================================

    def run(self):

        self._validate()

        summary = self.analyze()

        return summary
        # ======================================================
    # Detailed Generator Diagnostics
    # ======================================================

    def generator_diagnostics(
        self,
        generator_id: str
    ) -> dict:

        generator_df = self.df[
            self.df["generator_id"] == generator_id
        ]

        if len(generator_df) == 0:

            raise ValueError(
                f"Generator {generator_id} not found"
            )

        diffs = self._calculate_differences(
            generator_df
        )

        estimation = self._estimate_interval(
            diffs
        )

        return {

            "generator": generator_id,

            "detected_interval":

                estimation["interval"],

            "confidence":

                estimation["confidence"],

            "total_timestamp_differences":

                len(diffs),

            "dominant_cluster_size":

                len(estimation["cluster"]),

            "raw_difference_distribution":

                pd.Series(
                    diffs
                ).describe()

        }


    # ======================================================
    # Export Interval Dictionary
    # ======================================================

    def save_interval_dictionary(
        self,
        filepath: str
    ):

        if len(self.interval_dictionary) == 0:

            self.analyze()


        data = pd.DataFrame(

            list(
                self.interval_dictionary.items()
            ),

            columns=[

                "generator_id",

                "reporting_interval_seconds"

            ]

        )


        data.to_csv(

            filepath,

            index=False

        )


    # ======================================================
    # Compare Expected vs Detected
    # ======================================================

    def compare_generators(self):

        summary = self.get_summary()


        comparison = summary[

            [

                "Generator",

                "Reporting Interval (sec)",

                "Confidence (%)",

                "Stability"

            ]

        ]


        return comparison.sort_values(

            by="Confidence (%)",

            ascending=False

        )