"""
==============================================================
Feature Selector
==============================================================

Performs automatic feature selection for the Fuel Telemetry AI
pipeline before machine learning model training.

Features
--------
✔ Dataset validation
✔ Metadata removal
✔ Constant feature removal
✔ Missing value analysis
✔ Correlation analysis
✔ Low variance removal
✔ Feature statistics
✔ Report generation
✔ CSV export

Author:
Fuel Telemetry AI Project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================


@dataclass
class FeatureSelectorConfig:
    """
    Configuration for Feature Selector.
    """

    # Remove columns with missing ratio above this value
    missing_threshold: float = 0.50

    # Remove highly correlated features
    correlation_threshold: float = 0.95

    # Remove low variance features
    variance_threshold: float = 0.001

    # Remove constant / near-constant features
    remove_constant_features: bool = True

    # If one value occupies >= threshold of rows,
    # feature is considered constant.
    constant_threshold: float = 0.99

    # Optional ML target column
    target_column: Optional[str] = None

    # Metadata columns never used for ML
    metadata_columns: tuple = (
        "timestamp",
        "generator_id",
        "row_quality",
        "use_for_training",
    )


# ==========================================================
# Feature Selector
# ==========================================================


class FeatureSelector:
    """
    Automatic Feature Selection Pipeline.

    Pipeline
    --------
    Validate Dataset
            ↓
    Remove Metadata
            ↓
    Remove Constant Features
            ↓
    Remove High Missing Features
            ↓
    Remove Correlated Features
            ↓
    Remove Low Variance Features
            ↓
    Build Statistics
            ↓
    Export Reports
    """

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[FeatureSelectorConfig] = None
    ):

        self.df = df.copy()

        self.config = config or FeatureSelectorConfig()

        # Working dataframe
        self.selected_df = self.df.copy()

        # Report containers
        self.selected_features: List[str] = []

        self.removed_features: List[str] = []

        self.statistics: Dict = {}

    # ======================================================
    # Validation
    # ======================================================

    def validate(self):
        """
        Validate the input dataframe.
        """

        if self.df.empty:

            raise ValueError(
                "Input dataframe is empty."
            )

        if len(self.df.columns) == 0:

            raise ValueError(
                "Dataset contains no columns."
            )

    # ======================================================
    # Remove Metadata Columns
    # ======================================================

    def remove_metadata(self):
        """
        Remove columns that should never be used
        for machine learning.
        """

        removed = []

        for column in self.config.metadata_columns:

            if column in self.selected_df.columns:

                self.selected_df.drop(
                    columns=column,
                    inplace=True
                )

                removed.append(column)

        if removed:

            self.removed_features.extend(
                removed
            )

        return removed

    # ======================================================
    # Remove Constant Features
    # ======================================================

    def remove_constant_features(self):
        """
        Remove constant and near-constant features.

        A feature is considered constant when
        one value occupies more than the configured
        threshold of all observations.
        """

        if not self.config.remove_constant_features:

            return []

        removed = []

        for column in list(self.selected_df.columns):

            # Skip target column
            if column == self.config.target_column:
                continue

            values = self.selected_df[column]

            # Only numeric columns
            if not pd.api.types.is_numeric_dtype(values):
                continue

            dominant_ratio = (
                values
                .value_counts(
                    normalize=True,
                    dropna=False
                )
                .iloc[0]
            )

            if dominant_ratio >= self.config.constant_threshold:

                self.selected_df.drop(
                    columns=column,
                    inplace=True
                )

                removed.append(column)

        if removed:

            self.removed_features.extend(
                removed
            )

        return removed
        # ======================================================
    # Remove High Missing Features
    # ======================================================

    def remove_missing_features(self):
        """
        Remove features whose missing-value ratio exceeds
        the configured threshold.
        """

        removed = []

        missing_ratio = (
            self.selected_df
            .isna()
            .mean()
        )

        for column, ratio in missing_ratio.items():

            if column == self.config.target_column:
                continue

            if ratio > self.config.missing_threshold:

                removed.append(column)

        if removed:

            self.selected_df.drop(
                columns=removed,
                inplace=True
            )

            self.removed_features.extend(
                removed
            )

        return removed

    # ======================================================
    # Remove Highly Correlated Features
    # ======================================================

    def remove_correlated_features(self):
        """
        Remove redundant features that are highly
        correlated with other numeric features.
        """

        removed = []

        numeric_df = self.selected_df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:

            return removed

        correlation_matrix = (
            numeric_df
            .corr()
            .abs()
        )

        upper_triangle = correlation_matrix.where(

            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool)

        )

        for column in upper_triangle.columns:

            if column == self.config.target_column:
                continue

            if (

                upper_triangle[column]

                >

                self.config.correlation_threshold

            ).any():

                removed.append(column)

        if removed:

            self.selected_df.drop(
                columns=removed,
                inplace=True
            )

            self.removed_features.extend(
                removed
            )

        return removed

    # ======================================================
    # Remove Low Variance Features
    # ======================================================

    def remove_low_variance_features(self):
        """
        Remove numeric features having variance
        below the configured threshold.
        """

        removed = []

        numeric_df = self.selected_df.select_dtypes(
            include=np.number
        )

        if numeric_df.empty:

            return removed

        variance = numeric_df.var()

        for column, value in variance.items():

            if column == self.config.target_column:
                continue

            if value < self.config.variance_threshold:

                removed.append(column)

        if removed:

            self.selected_df.drop(
                columns=removed,
                inplace=True
            )

            self.removed_features.extend(
                removed
            )

        return removed
        # ======================================================
    # Build Statistics
    # ======================================================

    def build_statistics(self):
        """
        Build summary statistics for the feature
        selection process.
        """

        self.selected_features = list(
            self.selected_df.columns
        )

        removed = list(
            dict.fromkeys(
                self.removed_features
            )
        )

        numeric_df = self.selected_df.select_dtypes(
            include=np.number
        )

        if numeric_df.empty:

            variance = {}

        else:

            variance = (

                numeric_df
                .var()
                .round(6)
                .to_dict()

            )

        self.statistics = {

            "original_feature_count":

                len(self.df.columns),

            "selected_feature_count":

                len(self.selected_features),

            "removed_feature_count":

                len(removed),

            "selected_features":

                self.selected_features,

            "removed_features":

                removed,

            "missing_threshold":

                self.config.missing_threshold,

            "correlation_threshold":

                self.config.correlation_threshold,

            "variance_threshold":

                self.config.variance_threshold,

            "constant_threshold":

                self.config.constant_threshold,

            "missing_ratio":

                self.selected_df
                .isna()
                .mean()
                .round(4)
                .to_dict(),

            "variance":

                variance

        }

        return self.statistics

    # ======================================================
    # Selected Features
    # ======================================================

    def get_selected_features(self):
        """
        Return selected feature names.
        """

        return list(
            self.selected_df.columns
        )

    # ======================================================
    # Removed Features
    # ======================================================

    def get_removed_features(self):
        """
        Return removed feature names.
        """

        return list(

            dict.fromkeys(
                self.removed_features
            )

        )

    # ======================================================
    # Save Selected Dataset
    # ======================================================

    def save_selected_dataset(
        self,
        filepath: str
    ):
        """
        Save selected dataset.
        """

        self.selected_df.to_csv(
            filepath,
            index=False
        )

    # ======================================================
    # Save Reports
    # ======================================================

    def save_reports(
        self,
        report_directory: str

        
    ):
        """
        Save feature selection reports.
        """

        import json

        report_path = Path(
            report_directory
        )

        report_path.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame({

            "selected_feature":

                self.get_selected_features()

        }).to_csv(

            report_path /
            "selected_features.csv",

            index=False

        )

        pd.DataFrame({

            "removed_feature":

                self.get_removed_features()

        }).to_csv(

            report_path /
            "removed_features.csv",

            index=False

        )
        if not self.statistics:
         self.build_statistics()
        with open(

            report_path /
            "feature_statistics.json",

            "w"

        ) as file:

            json.dump(

                self.statistics,

                file,

                indent=4

            )

        return report_path
        # ======================================================
    # Print Report
    # ======================================================

    def print_report(self):
        """
        Print a summary of the feature selection process.
        """

        if not self.statistics:

            self.build_statistics()

        print()
        print("=" * 70)
        print("FEATURE SELECTION REPORT")
        print("=" * 70)

        print(
            f"Original Features : "
            f"{self.statistics['original_feature_count']}"
        )

        print(
            f"Selected Features : "
            f"{self.statistics['selected_feature_count']}"
        )

        print(
            f"Removed Features  : "
            f"{self.statistics['removed_feature_count']}"
        )

        print()

        if self.get_removed_features():

            print("Removed Features")
            print("-" * 70)

            for feature in self.get_removed_features():

                print(f"• {feature}")

        else:

            print("No features were removed.")

        print()

        print("Selected Features")
        print("-" * 70)

        for feature in self.get_selected_features():

            print(f"• {feature}")

        print()

    # ======================================================
    # Run Pipeline
    # ======================================================

    def run(self):
        """
        Execute the complete feature selection pipeline.

        Pipeline
        --------
        1. Validate dataset
        2. Remove metadata columns
        3. Remove constant features
        4. Remove high-missing features
        5. Remove highly correlated features
        6. Remove low-variance features
        7. Build statistics
        """

        self.validate()

        self.remove_metadata()

        self.remove_constant_features()

        self.remove_missing_features()

        self.remove_correlated_features()

        self.remove_low_variance_features()

        self.build_statistics()

        self.selected_features = self.get_selected_features()

        return self.selected_df