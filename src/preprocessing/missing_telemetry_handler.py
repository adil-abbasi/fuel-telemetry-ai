import pandas as pd
import numpy as np


class MissingTelemetryHandler:

    """
    Estimate short telemetry gaps while preserving
    original raw values.
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    # -------------------------------------------------

    def _prepare(self):

        self.df = self.df.sort_values(
            ["generator_id", "timestamp"]
        ).reset_index(drop=True)

        telemetry_columns = [
            "fuel_level_l",
            "current",
            "battery_voltage"
        ]

        # Create cleaned copies

        for col in telemetry_columns:

            self.df[f"{col}_clean"] = self.df[col]

        # Metadata

        self.df["is_estimated"] = False
        self.df["estimated_fields"] = ""
        self.df["estimation_method"] = ""

    # -------------------------------------------------

    def _gap_size(self):

        self.df["gap_minutes"] = (

            self.df.groupby("generator_id")["timestamp"]

            .diff()

            .dt.total_seconds()

            / 60

        )

    # -------------------------------------------------

    def _gap_category(self):

        conditions = [

            self.df["gap_minutes"] <= 2,

            (self.df["gap_minutes"] > 2)
            &
            (self.df["gap_minutes"] <= 5),

            (self.df["gap_minutes"] > 5)
            &
            (self.df["gap_minutes"] <= 30),

            self.df["gap_minutes"] > 30

        ]

        labels = [

            "normal",

            "short",

            "medium",

            "long"

        ]

        self.df["gap_type"] = np.select(

            conditions,

            labels,

            default="unknown"

        )

    # -------------------------------------------------

    def _estimate_column(self, column):

        clean_col = f"{column}_clean"

        grouped = self.df.groupby("generator_id")

        for generator, index in grouped.groups.items():

            temp = self.df.loc[index].copy()

            # Estimate only for normal gaps

            mask = (

                temp[column].isna()

                &

                temp["gap_type"].isin(

                    ["normal", "short"]

                )

            )

            temp.loc[:, clean_col] = temp[column].interpolate(

                method="linear",

                limit=2,

                limit_direction="both"

            )

            estimated_rows = mask & temp[clean_col].notna()

            self.df.loc[
                temp.index[estimated_rows],
                clean_col
            ] = temp.loc[
                estimated_rows,
                clean_col
            ]

            self.df.loc[
                temp.index[estimated_rows],
                "is_estimated"
            ] = True

            self.df.loc[
                temp.index[estimated_rows],
                "estimation_method"
            ] = "linear"

            self.df.loc[
                temp.index[estimated_rows],
                "estimated_fields"
            ] += column + ";"

    # -------------------------------------------------

    def _summary(self):

        report = pd.DataFrame({

            "Metric": [

                "Estimated Records",

                "Fuel Estimated",

                "Current Estimated",

                "Battery Estimated"

            ],

            "Value": [

                self.df["is_estimated"].sum(),

                self.df["fuel_level_l_clean"].notna().sum()
                -
                self.df["fuel_level_l"].notna().sum(),

                self.df["current_clean"].notna().sum()
                -
                self.df["current"].notna().sum(),

                self.df["battery_voltage_clean"].notna().sum()
                -
                self.df["battery_voltage"].notna().sum()

            ]

        })

        return report

    # -------------------------------------------------

    def run(self):

        self._prepare()

        self._gap_size()

        self._gap_category()

        self._estimate_column("fuel_level_l")

        self._estimate_column("current")

        self._estimate_column("battery_voltage")

        report = self._summary()

        return self.df, report