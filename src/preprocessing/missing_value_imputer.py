"""
==============================================================
Missing Value Imputer
==============================================================

Intelligent missing telemetry handling for generator data.

Input
-----
generator_state_estimated.csv

The dataset already contains:
- estimated_status
- running_probability
- estimated_confidence
- status_conflict

Features
--------
✓ Generator-wise processing
✓ Fuel interpolation
✓ Load estimation
✓ Battery estimation
✓ Status completion
✓ Fuel-rate recalculation
✓ Imputation confidence
✓ Reports

Author:
Fuel Telemetry AI Project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================


@dataclass
class MissingValueImputerConfig:

    generator_column: str = "generator_id"

    timestamp_column: str = "timestamp"

    fuel_column: str = "fuel_level_l"

    current_column: str = "current"

    battery_column: str = "battery_voltage"

    status_column: str = "status"

    estimated_status_column: str = "estimated_status"

    fuel_rate_column: str = "fuel_rate_lph"

    time_delta_column: str = "time_delta_sec"


    # interpolation settings

    interpolation_limit: int = 10


    # generator state

    stopped_state: str = "Stopped"


    # outputs

    create_flags: bool = True

    create_confidence: bool = True



# ==========================================================
# Missing Value Imputer
# ==========================================================


class MissingValueImputer:
    """
    Intelligent telemetry missing value processor.

    Pipeline
    --------

    Validate

        ↓

    Process each generator

        ↓

    Impute fuel

        ↓

    Impute current

        ↓

    Impute battery

        ↓

    Fix status

        ↓

    Recalculate fuel rate

        ↓

    Calculate confidence

        ↓

    Generate reports
    """


    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[MissingValueImputerConfig] = None,
    ):

        self.df = df.copy()

        self.config = (
            config
            or MissingValueImputerConfig()
        )


        self.output_df = self.df.copy()


        self.summary = {}
            # ======================================================
    # Validation
    # ======================================================

    def validate(self):
        """
        Validate required columns.
        """

        required_columns = [

            self.config.generator_column,

            self.config.timestamp_column,

            self.config.fuel_column,

            self.config.current_column,

            self.config.battery_column,

            self.config.status_column,

            self.config.estimated_status_column,

            self.config.time_delta_column,

        ]


        missing = [

            col

            for col in required_columns

            if col not in self.output_df.columns

        ]


        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )


        if self.output_df.empty:

            raise ValueError(
                "Input dataframe is empty."
            )


    # ======================================================
    # Generator-wise Processing
    # ======================================================

    def process_generators(self):
        """
        Process every generator separately.

        Important:
        Data from different generators should never
        influence each other.
        """

        processed = []


        grouped = self.output_df.groupby(
            self.config.generator_column,
            sort=False
        )


        for generator_id, generator_df in grouped:


            print(
                f"Processing {generator_id}"
            )


            generator_df = (
                generator_df
                .sort_values(
                    self.config.timestamp_column
                )
                .copy()
            )


            generator_df = (
                self.impute_fuel(
                    generator_df
                )
            )


            generator_df = (
                self.impute_current(
                    generator_df
                )
            )


            generator_df = (
                self.impute_battery(
                    generator_df
                )
            )


            generator_df = (
                self.impute_status(
                    generator_df
                )
            )


            generator_df = (
                self.recalculate_fuel_rate(
                    generator_df
                )
            )


            processed.append(
                generator_df
            )


        self.output_df = (
            pd.concat(processed)
            .sort_values(
                self.config.timestamp_column
            )
            .reset_index(drop=True)
        )


    # ======================================================
    # Fuel Imputation
    # ======================================================

    def impute_fuel(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing fuel values.

        Fuel is interpolated only for small gaps.
        Large missing periods remain missing.
        """


        if self.config.create_flags:

            df["fuel_imputed"] = (
                df[self.config.fuel_column]
                .isna()
            )


        df[self.config.fuel_column] = (

            df[self.config.fuel_column]

            .interpolate(

                method="linear",

                limit=self.config.interpolation_limit,

                limit_direction="both"

            )

        )


        return df



    # ======================================================
    # Current Imputation
    # ======================================================

    def impute_current(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate missing generator load.

        If generator is estimated stopped:
            current = 0

        Otherwise:
            interpolate
        """


        if self.config.create_flags:

            df["current_imputed"] = (
                df[self.config.current_column]
                .isna()
            )


        stopped = (

            df[self.config.estimated_status_column]

            == self.config.stopped_state

        )


        df.loc[

            stopped

            &

            df[self.config.current_column].isna(),

            self.config.current_column

        ] = 0



        df[self.config.current_column] = (

            df[self.config.current_column]

            .interpolate(

                method="linear",

                limit=self.config.interpolation_limit,

                limit_direction="both"

            )

        )


        return df



    # ======================================================
    # Battery Voltage Imputation
    # ======================================================

    def impute_battery(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate missing battery voltage.
        """


        if self.config.create_flags:

            df["battery_imputed"] = (

                df[self.config.battery_column]

                .isna()

            )


        stopped = (

            df[self.config.estimated_status_column]

            == self.config.stopped_state

        )


        df.loc[

            stopped

            &

            df[self.config.battery_column].isna(),

            self.config.battery_column

        ] = 0



        df[self.config.battery_column] = (

            df[self.config.battery_column]

            .interpolate(

                method="linear",

                limit=self.config.interpolation_limit,

                limit_direction="both"

            )

        )


        return df
        # ======================================================
    # Status Imputation
    # ======================================================

    def impute_status(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing status values.

        Uses estimated_status from
        Generator State Estimator.
        """


        if self.config.create_flags:

            df["status_imputed"] = (

                df[self.config.status_column]

                .isna()

            )


        missing_status = (

            df[self.config.status_column]

            .isna()

        )


        df.loc[

            missing_status,

            self.config.status_column

        ] = (

            df.loc[

                missing_status,

                self.config.estimated_status_column

            ]

        )


        return df



    # ======================================================
    # Fuel Rate Recalculation
    # ======================================================

    def recalculate_fuel_rate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Recalculate fuel consumption rate.

        Formula:

        fuel_rate_lph =
        -(fuel_change / time_seconds) * 3600


        Positive value means fuel consumed.
        """


        fuel_change = (

            df[self.config.fuel_column]

            .diff()

        )


        time_seconds = (

            df[self.config.time_delta_column]

        )


        rate = (

            -fuel_change

            /

            time_seconds

        ) * 3600



        rate = rate.replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan

        )



        df[

            self.config.fuel_rate_column

        ] = rate



        return df



    # ======================================================
    # Confidence Calculation
    # ======================================================

    def calculate_confidence(self):
        """
        Calculate confidence of imputed values.

        Higher score means:
        - less missing data
        - more reliable telemetry
        """


        if not self.config.create_confidence:

            return



        confidence = np.full(

            len(self.output_df),

            100.0

        )


        # Fuel missing penalty

        if "fuel_imputed" in self.output_df.columns:


            confidence -= (

                self.output_df["fuel_imputed"]

                .astype(float)

                *

                15

            )



        # Current missing penalty

        if "current_imputed" in self.output_df.columns:


            confidence -= (

                self.output_df["current_imputed"]

                .astype(float)

                *

                10

            )



        # Battery missing penalty

        if "battery_imputed" in self.output_df.columns:


            confidence -= (

                self.output_df["battery_imputed"]

                .astype(float)

                *

                10

            )



        # Status missing penalty

        if "status_imputed" in self.output_df.columns:


            confidence -= (

                self.output_df["status_imputed"]

                .astype(float)

                *

                10

            )



        # Status estimator confidence influence

        if "estimated_confidence" in self.output_df.columns:


            confidence = (

                confidence * 0.7

            ) + (

                self.output_df["estimated_confidence"]

                * 0.3

            )



        confidence = np.clip(

            confidence,

            0,

            100

        )



        self.output_df[

            "imputation_confidence"

        ] = confidence



        self.output_df[

            "imputation_confidence_level"

        ] = pd.cut(

            confidence,

            bins=[

                0,

                40,

                60,

                80,

                95,

                100

            ],

            labels=[

                "Very Low",

                "Low",

                "Medium",

                "High",

                "Very High"

            ],

            include_lowest=True

        )
            # ======================================================
    # Build Summary
    # ======================================================

    def build_summary(self):
        """
        Generate missing value imputation summary.
        """

        summary = {}


        columns = [

            self.config.fuel_column,

            self.config.current_column,

            self.config.battery_column,

            self.config.status_column,

        ]


        for column in columns:


            before = (

                self.df[column]

                .isna()

                .sum()

            )


            after = (

                self.output_df[column]

                .isna()

                .sum()

            )


            summary[column] = {

                "missing_before": int(before),

                "missing_after": int(after),

                "filled": int(before - after),

            }


        self.summary = summary


        return summary



    # ======================================================
    # Save Reports
    # ======================================================

    def save_reports(
        self,
        output_directory: str = "reports",
    ):
        """
        Save processed dataset and reports.
        """


        output_path = Path(
            output_directory
        )


        output_path.mkdir(

            parents=True,

            exist_ok=True

        )



        # Final dataset

        self.output_df.to_csv(

            output_path /

            "missing_value_imputed_dataset.csv",

            index=False

        )



        # Summary report

        rows = []


        for feature, values in self.summary.items():

            rows.append({

                "feature": feature,

                **values

            })


        pd.DataFrame(rows).to_csv(

            output_path /

            "missing_value_summary.csv",

            index=False

        )



        return str(output_path)



    # ======================================================
    # Print Report
    # ======================================================

    def print_report(self):
        """
        Display processing report.
        """


        print()

        print("=" * 70)

        print(
            "MISSING VALUE IMPUTATION REPORT"
        )

        print("=" * 70)


        print()


        total_before = 0

        total_after = 0



        for feature, values in self.summary.items():


            print(feature)


            print(
                f" Missing Before : {values['missing_before']}"
            )


            print(
                f" Missing After  : {values['missing_after']}"
            )


            print(
                f" Filled         : {values['filled']}"
            )


            print()


            total_before += values["missing_before"]

            total_after += values["missing_after"]



        print("-" * 70)


        print(
            f"Total Missing Before : {total_before}"
        )


        print(
            f"Total Missing After  : {total_after}"
        )


        print(
            f"Total Filled         : {total_before-total_after}"
        )


        print("-" * 70)



        if "imputation_confidence" in self.output_df.columns:


            print()

            print(

                "Average Imputation Confidence : "

                f"{self.output_df['imputation_confidence'].mean():.2f}%"

            )



            print()


            print(

                self.output_df[

                    "imputation_confidence_level"

                ]

                .value_counts()

                .sort_index()

            )



    # ======================================================
    # Public Methods
    # ======================================================

    def run(self):
        """
        Execute complete imputation pipeline.
        """


        self.validate()


        self.process_generators()


        self.calculate_confidence()


        self.build_summary()


        self.print_report()


        return self.output_df



    def get_summary(self):
        """
        Return summary dictionary.
        """

        return self.summary