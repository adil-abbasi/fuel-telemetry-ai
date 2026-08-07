"""
==============================================================
Telemetry Validator
==============================================================

Validates generator telemetry without modifying original data.

Detects:
- Impossible sensor values
- Sensor spikes
- Timestamp issues
- Data quality problems

Input:
generator_state_estimated.csv

Output:
validated_telemetry_dataset.csv

==============================================================
"""


from pathlib import Path

import pandas as pd

import numpy as np



# ==========================================================
# Configuration
# ==========================================================


class TelemetryValidationConfig:


    timestamp_column = "timestamp"

    generator_column = "generator_id"


    fuel_column = "fuel_level_l"

    current_column = "current"

    battery_column = "battery_voltage"



    # Expected ranges

    MIN_FUEL = 0

    MAX_FUEL = 5000



    MIN_CURRENT = 0

    MAX_CURRENT = 500



    MIN_BATTERY = 0

    MAX_BATTERY = 70



    # Timestamp

    MAX_TIME_GAP_SECONDS = 180



# ==========================================================
# Validator Class
# ==========================================================


class TelemetryValidator:


    def __init__(
        self,
        dataframe,
        config=None
    ):


        self.df = dataframe.copy()


        self.config = (

            config

            if config

            else TelemetryValidationConfig()

        )



    # ======================================================
    # Validate Required Columns
    # ======================================================


    def validate_columns(self):


        required = [

            self.config.timestamp_column,

            self.config.generator_column,

            self.config.fuel_column,

            self.config.current_column,

            self.config.battery_column,

        ]


        missing = [

            col

            for col in required

            if col not in self.df.columns

        ]


        if missing:

            raise ValueError(

                f"Missing columns: {missing}"

            )



        return True
        # ======================================================
    # Fuel Validation
    # ======================================================

    def validate_fuel(self):

        cfg = self.config


        self.df["fuel_invalid"] = False

        self.df["fuel_outlier"] = False



        # Impossible values

        invalid_mask = (

            (self.df[cfg.fuel_column] < cfg.MIN_FUEL)

            |

            (self.df[cfg.fuel_column] > cfg.MAX_FUEL)

        )


        self.df.loc[

            invalid_mask,

            "fuel_invalid"

        ] = True



        # Detect sudden fuel jumps

        self.df = self.df.sort_values(

            [

                cfg.generator_column,

                cfg.timestamp_column

            ]

        )



        fuel_change = (

            self.df

            .groupby(cfg.generator_column)

            [cfg.fuel_column]

            .diff()

        )


        self.df["fuel_change"] = fuel_change



        self.df.loc[

            fuel_change.abs() > 50,

            "fuel_outlier"

        ] = True



    # ======================================================
    # Current Validation
    # ======================================================


    def validate_current(self):


        cfg = self.config



        self.df["current_invalid"] = False

        self.df["current_outlier"] = False



        # Range validation

        invalid_current = (

            (self.df[cfg.current_column] < cfg.MIN_CURRENT)

            |

            (self.df[cfg.current_column] > cfg.MAX_CURRENT)

        )



        self.df.loc[

            invalid_current,

            "current_invalid"

        ] = True



        # Sudden load changes


        self.df = self.df.sort_values(

            [

                cfg.generator_column,

                cfg.timestamp_column

            ]

        )


        current_change = (

            self.df

            .groupby(cfg.generator_column)

            [cfg.current_column]

            .diff()

        )


        self.df["current_change"] = current_change



        self.df.loc[

            current_change.abs() > 150,

            "current_outlier"

        ] = True



    # ======================================================
    # Battery Validation
    # ======================================================


    def validate_battery(self):


        cfg = self.config



        self.df["battery_invalid"] = False



        invalid_voltage = (

            (self.df[cfg.battery_column] < cfg.MIN_BATTERY)

            |

            (self.df[cfg.battery_column] > cfg.MAX_BATTERY)

        )


        self.df.loc[

            invalid_voltage,

            "battery_invalid"

        ] = True


        # ======================================================
    # Sensor Consistency Validation
    # ======================================================

    def validate_sensor_consistency(self):

        cfg = self.config


        # --------------------------------------------------
        # Current exists but battery charging missing
        # --------------------------------------------------

        self.df["battery_current_mismatch"] = False


        mismatch_condition = (

            (self.df[cfg.current_column] > 20)

            &

            (

                self.df[cfg.battery_column]

                < 10

            )

        )


        self.df.loc[

            mismatch_condition,

            "battery_current_mismatch"

        ] = True



        # --------------------------------------------------
        # Estimated state vs sensors
        # --------------------------------------------------

        self.df["state_sensor_mismatch"] = False



        if "estimated_status" in self.df.columns:


            running_states = [

                "Running (Low Load)",

                "Running (Medium Load)",

                "Running (High Load)"

            ]



            # Estimated running but no current

            mismatch_running = (

                self.df[

                    "estimated_status"

                ]

                .isin(running_states)

                &

                (

                    self.df[

                        cfg.current_column

                    ]

                    <= 5

                )

            )



            self.df.loc[

                mismatch_running,

                "state_sensor_mismatch"

            ] = True



            # Estimated stopped but strong load exists

            mismatch_stopped = (

                self.df[

                    "estimated_status"

                ]

                .isin(

                    [

                        "Stopped",

                        "Idle"

                    ]

                )

                &

                (

                    self.df[

                        cfg.current_column

                    ]

                    > 30

                )

            )



            self.df.loc[

                mismatch_stopped,

                "state_sensor_mismatch"

            ] = True
    # ======================================================
    # Timestamp Validation
    # ======================================================


    def validate_timestamps(self):


        cfg = self.config



        self.df["timestamp_duplicate"] = (

            self.df

            .duplicated(

                subset=[

                    cfg.generator_column,

                    cfg.timestamp_column

                ],

                keep=False

            )

        )



        self.df = self.df.sort_values(

            [

                cfg.generator_column,

                cfg.timestamp_column

            ]

        )



        time_gap = (

            self.df

            .groupby(cfg.generator_column)

            [cfg.timestamp_column]

            .diff()

            .dt.total_seconds()

        )



        self.df["timestamp_gap_seconds"] = time_gap



        self.df["timestamp_gap"] = (

            time_gap >

            cfg.MAX_TIME_GAP_SECONDS

        )
            # ======================================================
    # Telemetry Quality Score
    # ======================================================

    def calculate_quality_score(self):


        score = np.ones(

            len(self.df)

        ) * 100



        # Reduce score for invalid sensors

        score -= (

            self.df["fuel_invalid"]

            * 20

        )


        score -= (

            self.df["current_invalid"]

            * 20

        )


        score -= (

            self.df["battery_invalid"]

            * 15

        )
        score -= (

            self.df["battery_current_mismatch"]

            * 5

        )


        score -= (

            self.df["state_sensor_mismatch"]

            * 10

        )


        # Reduce for outliers

        score -= (

            self.df["fuel_outlier"]

            * 15

        )


        score -= (

            self.df["current_outlier"]

            * 15

        )



        # Timestamp problems

        score -= (

            self.df["timestamp_duplicate"]

            * 10

        )


        score -= (

            self.df["timestamp_gap"]

            * 15

        )



        # Keep range 0-100

        score = np.clip(

            score,

            0,

            100

        )



        self.df[

            "telemetry_quality_score"

        ] = score.round(2)



    # ======================================================
    # Validation Reason
    # ======================================================


    def generate_validation_reason(self):


        reasons = []



        for _, row in self.df.iterrows():


            row_reason = []



            if row["fuel_invalid"]:

                row_reason.append(

                    "Invalid fuel level"

                )


            if row["fuel_outlier"]:

                row_reason.append(

                    "Fuel spike detected"

                )


            if row["current_invalid"]:

                row_reason.append(

                    "Invalid current"

                )


            if row["current_outlier"]:

                row_reason.append(

                    "Current spike detected"

                )


            if row["battery_invalid"]:

                row_reason.append(

                    "Invalid battery voltage"

                )

            
        
            if row["battery_current_mismatch"]:

                row_reason.append(
                    "Current load detected but battery charging signal missing"
                )


            if row["state_sensor_mismatch"]:            

                row_reason.append(
                    "Estimated state conflicts with sensor behavior"
                )
            if row["timestamp_duplicate"]:

                row_reason.append(

                    "Duplicate telemetry"

                )


            if row["timestamp_gap"]:

                row_reason.append(

                    "Telemetry delay detected"

                )



            if len(row_reason) == 0:

                row_reason.append(

                    "Telemetry healthy"

                )



            reasons.append(

                "; ".join(row_reason)

            )



        self.df[

            "validation_reason"

        ] = reasons



    # ======================================================
    # Run Full Validation
    # ======================================================


    def run(self):


        print()

        print(

            "Running Telemetry Validation..."

        )


        self.validate_columns()



        self.df[

            self.config.timestamp_column

        ] = pd.to_datetime(

            self.df[

                self.config.timestamp_column

            ],

            errors="coerce"

        )



        self.validate_fuel()


        self.validate_current()


        self.validate_battery()

        self.validate_sensor_consistency()


        self.validate_timestamps()



        self.calculate_quality_score()


        self.generate_validation_reason()



        return self.df