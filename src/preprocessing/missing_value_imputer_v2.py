"""
==============================================================
Missing Value Imputer V2
==============================================================

Smart telemetry imputation using:

- Generator state estimation
- Temporal interpolation
- Sensor validation
- Confidence scoring

Input:
validated_telemetry_dataset.csv


Output:
imputed_telemetry_dataset.csv

==============================================================
"""


from pathlib import Path

import numpy as np

import pandas as pd



# ==========================================================
# Configuration
# ==========================================================


class ImputerConfig:


    generator_column = "generator_id"

    timestamp_column = "timestamp"


    fuel_column = "fuel_level_l"

    current_column = "current"

    battery_column = "battery_voltage"

    status_column = "status"


    estimated_state_column = (

        "estimated_status"

    )



    interpolation_limit = 10



# ==========================================================
# Missing Value Imputer
# ==========================================================


class MissingValueImputerV2:


    def __init__(

        self,

        dataframe,

        config=None

    ):


        self.df = dataframe.copy()


        self.config = (

            config

            if config

            else ImputerConfig()

        )



        self.imputation_log = []



    # ======================================================
    # Prepare Data
    # ======================================================


    def prepare(self):


        cfg = self.config



        self.df[

            cfg.timestamp_column

        ] = pd.to_datetime(

            self.df[

                cfg.timestamp_column

            ],

            errors="coerce"

        )



        self.df = self.df.sort_values(

            [

                cfg.generator_column,

                cfg.timestamp_column

            ]

        )



        return self.df
        # ======================================================
    # Generic Interpolation Helper
    # ======================================================

    def interpolate_sensor(
        self,
        column
    ):

        cfg = self.config


        before_missing = (

            self.df[column]

            .isna()

            .sum()

        )


        # Interpolate separately for each generator

        self.df[column] = (

            self.df

            .groupby(

                cfg.generator_column

            )[column]

            .transform(

                lambda x:

                x.interpolate(

                    method="linear",

                    limit=

                    cfg.interpolation_limit,

                    limit_direction="both"

                )

            )

        )


        after_missing = (

            self.df[column]

            .isna()

            .sum()

        )


        filled = (

            before_missing

            -

            after_missing

        )


        return {

            "before": before_missing,

            "after": after_missing,

            "filled": filled

        }



    # ======================================================
    # Fuel Imputation
    # ======================================================


    def impute_fuel(self):


        column = self.config.fuel_column


        self.df["fuel_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "fuel_imputed"

        ] = True




    # ======================================================
    # Current Imputation
    # ======================================================


    def impute_current(self):


        column = self.config.current_column


        self.df["current_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "current_imputed"

        ] = True




    # ======================================================
    # Battery Voltage Imputation
    # ======================================================


    def impute_battery(self):


        column = self.config.battery_column


        self.df["battery_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "battery_imputed"

        ] = True
            # ======================================================
    # Generic Interpolation Helper
    # ======================================================

    def interpolate_sensor(
        self,
        column
    ):

        cfg = self.config


        before_missing = (

            self.df[column]

            .isna()

            .sum()

        )


        # Interpolate separately for each generator

        self.df[column] = (

            self.df

            .groupby(

                cfg.generator_column

            )[column]

            .transform(

                lambda x:

                x.interpolate(

                    method="linear",

                    limit=

                    cfg.interpolation_limit,

                    limit_direction="both"

                )

            )

        )


        after_missing = (

            self.df[column]

            .isna()

            .sum()

        )


        filled = (

            before_missing

            -

            after_missing

        )


        return {

            "before": before_missing,

            "after": after_missing,

            "filled": filled

        }



    # ======================================================
    # Fuel Imputation
    # ======================================================


    def impute_fuel(self):


        column = self.config.fuel_column


        self.df["fuel_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "fuel_imputed"

        ] = True




    # ======================================================
    # Current Imputation
    # ======================================================


    def impute_current(self):


        column = self.config.current_column


        self.df["current_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "current_imputed"

        ] = True




    # ======================================================
    # Battery Voltage Imputation
    # ======================================================

    
    def impute_battery(self):


        column = self.config.battery_column


        self.df["battery_imputed"] = False



        missing_before = (

            self.df[column]

            .isna()

        )



        self.interpolate_sensor(

            column

        )



        self.df.loc[

            missing_before

            &

            self.df[column].notna(),

            "battery_imputed"

        ] = True

        # ======================================================
    # Status Imputation
    # ======================================================

    def impute_status(self):

        if "estimated_status" not in self.df.columns:

            self.df["status_imputed"] = False
            return


        self.df["status_imputed"] = False


        missing_status = (

            self.df["status"].isna()

            |

            (

                self.df["status"]

                .astype(str)

                .str.lower()

                .isin(

                    [

                        "unknown",

                        "nan",

                        "none"

                    ]

                )

            )

        )


        mapping = {

            "Running (Low Load)": "running",

            "Running (Medium Load)": "running",

            "Running (High Load)": "running",

            "Idle": "stopped",

            "Stopped": "stopped",

            "Unknown": "unknown"

        }


        self.df.loc[
            missing_status,
            "status"
        ] = (

            self.df.loc[
                missing_status,
                "estimated_status"
            ]

            .map(mapping)

            .fillna("unknown")

        )


        self.df.loc[
            missing_status,
            "status_imputed"
        ] = True



    # ======================================================
    # Confidence Calculation
    # ======================================================

    def calculate_imputation_confidence(self):


        confidence = np.ones(
            len(self.df)
        ) * 100



        confidence -= (

            self.df["fuel_imputed"]

            * 20

        )


        confidence -= (

            self.df["current_imputed"]

            * 20

        )


        confidence -= (

            self.df["battery_imputed"]

            * 20

        )


        confidence -= (

            self.df["status_imputed"]

            * 15

        )


        if "telemetry_quality_score" in self.df.columns:


            confidence *= (

                self.df[

                    "telemetry_quality_score"

                ]

                /100

            )



        self.df[

            "imputation_confidence"

        ] = confidence.clip(

            0,

            100

        ).round(2)



    # ======================================================
    # Confidence Level
    # ======================================================

    def confidence_level(self):


        def classify(x):

            if x >= 90:

                return "Very High"

            elif x >= 75:

                return "High"

            elif x >= 50:

                return "Medium"

            elif x >= 25:

                return "Low"

            else:

                return "Very Low"



        self.df[

            "imputation_confidence_level"

        ] = (

            self.df[

                "imputation_confidence"

            ]

            .apply(classify)

        )

    def run(self):


        print()

        print("=" * 70)
        print("RUNNING MISSING VALUE IMPUTER V2")
        print("=" * 70)



        self.prepare()


        print(
            "Imputing fuel..."
        )

        self.impute_fuel()



        print(
            "Imputing current..."
        )

        self.impute_current()



        print(
            "Imputing battery..."
        )

        self.impute_battery()



        print(
            "Imputing status..."
        )

        self.impute_status()



        print(
            "Calculating confidence..."
        )

        self.calculate_imputation_confidence()



        self.confidence_level()



        return self.df