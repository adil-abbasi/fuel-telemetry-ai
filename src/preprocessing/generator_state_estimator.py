"""
==============================================================
Generator State Estimator
==============================================================

Intelligent generator operating state estimation from telemetry.

Purpose
-------
The raw generator status field is considered unreliable.

This module estimates the true generator state using:

- Current load
- Battery voltage
- Fuel consumption trend
- Fuel level change
- Sensor agreement

Output
------
Adds:

estimated_status
running_probability
estimated_confidence
confidence_level
status_conflict
estimation_reason
sensor_votes

Author:
Fuel Telemetry AI Project
"""


from __future__ import annotations


from dataclasses import dataclass
from typing import Optional


import numpy as np
import pandas as pd



# ==========================================================
# Configuration
# ==========================================================


@dataclass
class GeneratorStateEstimatorConfig:
    """
    Configuration for generator state estimation.
    """


    # Column names

    current_column: str = "current"

    battery_column: str = "battery_voltage"

    fuel_column: str = "fuel_level_l"

    status_column: str = "status"

    generator_column: str = "generator_id"

    timestamp_column: str = "timestamp"



    # Current thresholds

    running_current_threshold: float = 30.0


    low_load_current: float = 80.0

    medium_load_current: float = 150.0



    # Battery thresholds

    charging_voltage_threshold: float = 48.0



    # Fuel movement

    fuel_consumption_threshold: float = 0.02



    # Confidence weights

    current_weight: float = 0.50

    battery_weight: float = 0.30

    fuel_weight: float = 0.20



    # Conflict detection

    conflict_penalty: float = 20.0




# ==========================================================
# Generator State Estimator
# ==========================================================


class GeneratorStateEstimator:
    """
    Estimate actual generator operational state.

    Pipeline
    --------

    Validate Dataset

            ↓

    Generate sensor votes

            ↓

    Calculate running probability

            ↓

    Classify generator state

            ↓

    Detect conflicts

            ↓

    Generate confidence


    """


    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[
            GeneratorStateEstimatorConfig
        ] = None,

    ):

        self.df = df.copy()


        self.config = (

            config

            or GeneratorStateEstimatorConfig()

        )


        self.output_df = self.df.copy()
            # ======================================================
    # Validation
    # ======================================================

    def validate(self):
        """
        Validate required telemetry columns.
        """

        required_columns = [

            self.config.current_column,

            self.config.battery_column,

            self.config.fuel_column,

            self.config.status_column,

            self.config.generator_column,

        ]


        missing = [

            column

            for column in required_columns

            if column not in self.output_df.columns

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
    # Current Sensor Vote
    # ======================================================

    def evaluate_current(
        self,
        row,
    ):
        """
        Evaluate generator load using current.

        Current is the strongest indicator.
        """


        current = row[
            self.config.current_column
        ]


        if pd.isna(current):

            return {

                "vote": "Unknown",

                "score": 0,

                "reason": "Current unavailable"

            }



        if current >= self.config.medium_load_current:

            return {

                "vote": "Running",

                "score": 1.0,

                "reason":

                f"Current {current:.2f}A indicates high load"

            }



        if current >= self.config.low_load_current:

            return {

                "vote": "Running",

                "score": 0.9,

                "reason":

                f"Current {current:.2f}A indicates medium load"

            }



        if current >= self.config.running_current_threshold:

            return {

                "vote": "Running",

                "score": 0.7,

                "reason":

                f"Current {current:.2f}A indicates low load"

            }



        return {

            "vote": "Stopped",

            "score": 0.6,

            "reason":

            f"Current {current:.2f}A below running threshold"

        }



    # ======================================================
    # Battery Sensor Vote
    # ======================================================

    def evaluate_battery(
        self,
        row,
    ):
        """
        Evaluate battery charging behaviour.
        """


        voltage = row[
            self.config.battery_column
        ]


        if pd.isna(voltage):

            return {

                "vote": "Unknown",

                "score": 0,

                "reason":

                "Battery voltage unavailable"

            }



        if voltage >= self.config.charging_voltage_threshold:

            return {

                "vote": "Running",

                "score": 1.0,

                "reason":

                f"Battery {voltage:.2f}V indicates charging"

            }



        return {

    "vote": "Unknown",

    "score": 0,

    "reason":

    f"Battery {voltage:.2f}V not indicating charging"

}



    # ======================================================
    # Fuel Trend Vote
    # ======================================================

    def evaluate_fuel(
        self,
        row,
    ):
        """
        Evaluate fuel consumption.

        Fuel decreasing indicates generator usage.
        """


        if "fuel_delta" not in row.index:

            return {

                "vote": "Unknown",

                "score": 0,

                "reason":

                "Fuel delta unavailable"

            }



        fuel_delta = row["fuel_delta"]


        if pd.isna(fuel_delta):

            return {

                "vote": "Unknown",

                "score": 0,

                "reason":

                "Fuel change unavailable"

            }



        if (

            fuel_delta

            <

            -self.config.fuel_consumption_threshold

        ):

            return {

                "vote": "Running",

                "score": 1.0,

                "reason":

                "Fuel level decreasing"

            }



        return {

            "vote": "Stopped",

            "score": 0.4,

            "reason":

            "No fuel consumption detected"

        }



    # ======================================================
    # Generate Sensor Votes
    # ======================================================

    def generate_sensor_votes(
        self,
        row,
    ):
        """
        Collect all sensor decisions.
        """


        current_vote = self.evaluate_current(row)


        battery_vote = self.evaluate_battery(row)


        fuel_vote = self.evaluate_fuel(row)



        votes = {

            "current": current_vote,

            "battery_voltage": battery_vote,

            "fuel": fuel_vote,

        }


        return votes
        # ======================================================
    # Combine Sensor Votes
    # ======================================================

    def calculate_running_probability(
        self,
        votes,
    ):
        """
        Calculate probability that generator is running.

        Weighted combination:

        Current        50%
        Battery        30%
        Fuel           20%
        """


        probability = 0.0


        total_weight = 0.0



        sensor_weights = {

            "current":
                self.config.current_weight,

            "battery_voltage":
                self.config.battery_weight,

            "fuel":
                self.config.fuel_weight,

        }



        for sensor, weight in sensor_weights.items():

            vote = votes[sensor]


            if vote["vote"] == "Unknown":

                continue


            total_weight += weight



            if vote["vote"] == "Running":

                probability += (
                    vote["score"] * weight
                )



        if total_weight == 0:

            return 0.5



        return round(

            probability / total_weight,

            3

        )



    # ======================================================
    # Classify Generator State
    # ======================================================

    def classify_state(
        self,
        row,
        probability,
    ):
        """
        Convert probability into generator state.
        """


        current = row[
            self.config.current_column
        ]



        if pd.isna(current):

            current = 0



        if probability >= 0.75:


            if current >= self.config.medium_load_current:

                return "Running (High Load)"



            elif current >= self.config.low_load_current:

                return "Running (Medium Load)"



            else:

                return "Running (Low Load)"



        elif probability >= 0.55:

            if current >= self.config.running_current_threshold:

                return "Running (Low Load)"

            else:

                return "Idle"

        else:

            return "Unknown"



    # ======================================================
    # Confidence Calculation
    # ======================================================

    def calculate_confidence(
        self,
        votes,
        probability,
    ):
        """
        Calculate estimation confidence.

        Confidence depends on:
        - sensor availability
        - agreement
        - probability strength
        """


        available_votes = 0

        agreement = 0



        running_votes = 0

        stopped_votes = 0



        for sensor, vote in votes.items():


            if vote["vote"] == "Unknown":

                continue



            available_votes += 1



            if vote["vote"] == "Running":

                running_votes += 1


            elif vote["vote"] == "Stopped":

                stopped_votes += 1



        if available_votes == 0:

            return 0.0



        if running_votes > stopped_votes:

            agreement = (

                running_votes

                /

                available_votes

            )


        else:

            agreement = (

                stopped_votes

                /

                available_votes

            )



        probability_strength = abs(

            probability - 0.5

        ) * 2



        confidence = (

            agreement * 60

        ) + (

            probability_strength * 40

        )



        return round(

            min(confidence, 100),

            2

        )



    # ======================================================
    # Confidence Level
    # ======================================================

    def confidence_level(
        self,
        confidence,
    ):
        """
        Convert confidence number to category.
        """


        if confidence >= 90:

            return "Very High"


        if confidence >= 75:

            return "High"


        if confidence >= 50:

            return "Medium"


        if confidence >= 25:

            return "Low"


        return "Very Low"



    # ======================================================
    # Detect Status Conflict
    # ======================================================

    def detect_conflict(
        self,
        original_status,
        estimated_status,
    ):
        """
        Detect disagreement between
        telemetry status and estimated state.
        """


        if pd.isna(original_status):

            return False



        if str(original_status).lower() == "running":

            original_running = True

        else:

            original_running = False



        estimated_running = (

            "Running"

            in estimated_status

        )



        return (

            original_running

            !=

            estimated_running

        )



    # ======================================================
    # Build Estimation Reason
    # ======================================================

    def build_reason(
        self,
        votes,
    ):
        """
        Create human-readable explanation.
        """


        reasons = []



        for sensor, vote in votes.items():


            if vote["vote"] != "Unknown":

                reasons.append(

                    vote["reason"]

                )



        if not reasons:

            return "No telemetry evidence available."



        return ". ".join(reasons) + "."
        # ======================================================
    # Process Single Row
    # ======================================================

    def process_row(
        self,
        row,
    ):
        """
        Estimate generator state for one telemetry row.
        """


        votes = self.generate_sensor_votes(row)


        probability = self.calculate_running_probability(
            votes
        )


        estimated_state = self.classify_state(
            row,
            probability
        )


        confidence = self.calculate_confidence(
            votes,
            probability
        )


        conflict = self.detect_conflict(
            row[self.config.status_column],
            estimated_state
        )


        reason = self.build_reason(
            votes
        )


        sensor_vote_summary = {

            sensor: vote["vote"]

            for sensor, vote in votes.items()

        }


        return pd.Series({

            "estimated_status":
                estimated_state,


            "running_probability":
                probability,


            "estimated_confidence":
                confidence,


            "confidence_level":
                self.confidence_level(
                    confidence
                ),


            "status_conflict":
                conflict,


            "estimation_reason":
                reason,


            "sensor_votes":
                str(
                    sensor_vote_summary
                ),

        })



    # ======================================================
    # Estimate Dataset
    # ======================================================

    def estimate_states(self):
        """
        Apply estimator on complete dataframe.
        """


        results = self.output_df.apply(

            self.process_row,

            axis=1

        )


        self.output_df = pd.concat(

            [

                self.output_df,

                results

            ],

            axis=1

        )


        return self.output_df



    # ======================================================
    # Generate Summary
    # ======================================================

    def generate_summary(self):
        """
        Generate estimation statistics.
        """


        summary = {


            "total_rows":

                len(self.output_df),



            "status_conflicts":

                int(

                    self.output_df[

                        "status_conflict"

                    ]

                    .sum()

                ),



            "average_confidence":

                round(

                    self.output_df[

                        "estimated_confidence"

                    ]

                    .mean(),

                    2

                ),



            "estimated_state_distribution":

                self.output_df[

                    "estimated_status"

                ]

                .value_counts()

                .to_dict(),



            "confidence_distribution":

                self.output_df[

                    "confidence_level"

                ]

                .value_counts()

                .to_dict(),

        }


        return summary



    # ======================================================
    # Save Results
    # ======================================================

    def save_results(
        self,
        output_path="data/processed/generator_state_estimated.csv",
    ):
        """
        Save estimated telemetry dataset.
        """


        path = Path(output_path)


        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )


        self.output_df.to_csv(

            path,

            index=False

        )


        return str(path)



    # ======================================================
    # Print Report
    # ======================================================

    def print_report(self):
        """
        Display estimator report.
        """


        summary = self.generate_summary()



        print()

        print("=" * 70)

        print(
            "GENERATOR STATE ESTIMATION REPORT"
        )

        print("=" * 70)



        print()

        print(

            f"Rows Processed : "

            f"{summary['total_rows']}"

        )


        print(

            f"Telemetry Conflicts : "

            f"{summary['status_conflicts']}"

        )


        print(

            f"Average Confidence : "

            f"{summary['average_confidence']}%"

        )


        print()

        print(
            "Estimated State Distribution"
        )


        for state, count in summary[

            "estimated_state_distribution"

        ].items():


            print(

                f"{state:<30}"

                f"{count}"

            )



        print()

        print(
            "Confidence Distribution"
        )


        for level, count in summary[

            "confidence_distribution"

        ].items():


            print(

                f"{level:<30}"

                f"{count}"

            )



    # ======================================================
    # Public Run Method
    # ======================================================

    def run(
        self,
        save=True,
    ):
        """
        Complete estimation pipeline.
        """


        self.validate()


        self.output_df = (

            self.output_df

            .sort_values(

                self.config.timestamp_column

            )

            .reset_index(drop=True)

        )



        self.estimate_states()



        self.print_report()



        if save:

            self.save_results()



        return self.output_df