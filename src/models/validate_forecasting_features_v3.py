"""
V3 FORECASTING FEATURE VALIDATOR

Purpose
-------
Validate that the V3 feature-engineered dataset:

1. Preserves the exact source row count.
2. Preserves every original telemetry + target row.
3. Preserves target_fuel_3h exactly.
4. Preserves duplicate generator/timestamp structure.
5. Contains no duplicate column names.
6. Contains no infinite numeric values.
7. Does not contain obvious future-value leakage.
8. Contains all required V3 features.
9. Preserves chronological ordering within each generator.
10. Does not require generator_id + timestamp to be unique.

Important
---------
The source dataset contains duplicate
(generator_id, timestamp) rows.

Therefore:

- generator_id + timestamp is NOT a unique row key.
- Source row order is NOT used for matching.
- Duplicate rows are preserved.
- Validation uses an order-independent canonical multiset
  comparison of the original telemetry + target columns.

Target metadata
---------------
The V3 feature builder may legitimately contain:

    target_timestamp
    target_lookup_time
    target_time_difference_minutes

These columns describe target alignment / target availability.

They are NOT future telemetry values and are therefore allowed
by this validator.

IMPORTANT MODELING NOTE
-----------------------
Although target metadata is allowed by this validator, target-derived
columns must NOT be used as input features during model training.

Examples:

    target_timestamp
    target_lookup_time
    target_time_difference_minutes
    target_negative
    target_negative_024
    target_valid_for_regression

These are validation/target-diagnostic columns, not model predictors.

Validator version
-----------------
2026-08-10-order-independent-v5
"""

from pathlib import Path

import numpy as np
import pandas as pd


V3_VALIDATOR_VERSION = "2026-08-10-order-independent-v5"


# ============================================================
# PATHS
# ============================================================

SOURCE_FILE = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)

V3_FILE = Path(
    "data/processed/fuel_forecasting_features_v3.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target_fuel_3h"

TELEMETRY_COLUMNS = [
    "timestamp",
    "generator_id",
    "site_name",
    "status",
    "fuel_level_l",
    "current",
    "battery_voltage",
]

EPSILON = 1e-9


# ============================================================
# REQUIRED V3 FEATURES
# ============================================================

REQUIRED_FEATURES = [

    # Time
    "hour",
    "minute",
    "weekday",
    "day",
    "month",
    "is_weekend",

    # Basic fuel behavior
    "fuel_delta",
    "time_delta_sec",
    "fuel_rate_lps",
    "fuel_rate_lph",

    # Sensor changes
    "current_delta",
    "voltage_delta",

    # Sensor validity
    "fuel_invalid",
    "current_invalid",
    "battery_invalid",

    # Outliers
    "fuel_outlier",
    "current_outlier",

    # Timestamp gaps
    "timestamp_gap_seconds",
    "timestamp_gap",

    # Fuel changes
    "fuel_change_15min",
    "fuel_change_30min",
    "fuel_change_60min",

    # Fuel refill/drop
    "fuel_refill_signal",
    "fuel_drop_signal",

    # Fuel rolling means
    "fuel_mean_5min",
    "fuel_mean_15min",
    "fuel_mean_30min",
    "fuel_mean_60min",

    # Fuel rolling standard deviation
    "fuel_std_5min",
    "fuel_std_15min",
    "fuel_std_30min",
    "fuel_std_60min",

    # Current rolling means
    "current_mean_5min",
    "current_mean_15min",
    "current_mean_30min",
    "current_mean_60min",

    # Current rolling standard deviation
    "current_std_5min",
    "current_std_15min",
    "current_std_30min",
    "current_std_60min",

    # Battery rolling means
    "battery_mean_5min",
    "battery_mean_15min",
    "battery_mean_30min",
    "battery_mean_60min",

    # Battery rolling standard deviation
    "battery_std_5min",
    "battery_std_15min",
    "battery_std_30min",
    "battery_std_60min",

    # Fuel trends
    "fuel_trend_15min",
    "fuel_trend_30min",
    "fuel_trend_60min",

    # Operating state
    "battery_zero",
    "fuel_zero_or_negative",
    "current_active",
    "sensor_state_conflict",

    # Target diagnostics
    "target_negative",
    "target_negative_024",
    "target_valid_for_regression",
]


# ============================================================
# TARGET-DERIVED METADATA
# ============================================================

ALLOWED_TARGET_METADATA = {
    "target_timestamp",
    "target_lookup_time",
    "target_time_difference_minutes",
    "target_negative",
    "target_negative_024",
    "target_valid_for_regression",
}


# ============================================================
# EXPLICITLY FORBIDDEN FUTURE-VALUE COLUMNS
# ============================================================

FORBIDDEN_FUTURE_COLUMNS = {
    "future_fuel",
    "future_fuel_level",
    "future_fuel_value",
    "future_fuel_l",
    "future_fuel_delta",
    "future_target",
    "future_target_value",
    "future_current",
    "future_battery_voltage",
    "future_status",
    "future_timestamp",
    "target_future_fuel",
    "target_future_value",
    "target_future_fuel_level",
    "target_value_future",
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path, name):

    print(f"\nLoading {name}...")

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    print(
        f"{name} rows: {len(df):,}"
    )

    return df


# ============================================================
# REQUIRED SOURCE / TARGET COLUMNS
# ============================================================

def validate_required_source_columns(source, v3):

    required = TELEMETRY_COLUMNS + [
        TARGET_COLUMN
    ]

    missing_source = [
        column
        for column in required
        if column not in source.columns
    ]

    missing_v3 = [
        column
        for column in required
        if column not in v3.columns
    ]

    if missing_source:
        raise AssertionError(
            "Missing required source columns: "
            + str(missing_source)
        )

    if missing_v3:
        raise AssertionError(
            "Missing required V3 columns: "
            + str(missing_v3)
        )

    print(
        "[PASS] All required source/target columns exist."
    )


# ============================================================
# ROW COUNT
# ============================================================

def validate_row_count(source, v3):

    source_rows = len(source)
    v3_rows = len(v3)

    print(
        f"\nSource rows: {source_rows:,}"
    )

    print(
        f"V3 rows:     {v3_rows:,}"
    )

    print(
        f"Difference:  {v3_rows - source_rows:,}"
    )

    if source_rows != v3_rows:
        raise AssertionError(
            "V3 row count does not match source."
        )

    print(
        "[PASS] Row count preserved."
    )


# ============================================================
# COLUMN NAME VALIDATION
# ============================================================

def validate_column_names(v3):

    duplicate_count = int(
        v3.columns.duplicated().sum()
    )

    print(
        "\nDuplicate column names:",
        duplicate_count,
    )

    if duplicate_count:

        duplicates = (
            v3.columns[
                v3.columns.duplicated()
            ]
            .tolist()
        )

        raise AssertionError(
            "Duplicate column names detected: "
            + str(duplicates)
        )

    print(
        "[PASS] Column names are unique."
    )


# ============================================================
# SOURCE COLUMN PRESERVATION
# ============================================================

def validate_source_columns(source, v3):

    print_section(
        "SOURCE COLUMN PRESERVATION"
    )

    missing_source_columns = [
        column
        for column in source.columns
        if column not in v3.columns
    ]

    print(
        "Source columns:",
        len(source.columns),
    )

    print(
        "V3 columns:",
        len(v3.columns),
    )

    if missing_source_columns:

        print(
            "\nSource columns missing from V3:"
        )

        for column in missing_source_columns:
            print(
                f"  [MISSING] {column}"
            )

        raise AssertionError(
            "One or more source columns are missing "
            "from V3."
        )

    print(
        "[PASS] All source columns preserved."
    )


# ============================================================
# DUPLICATE STRUCTURE
# ============================================================

def duplicate_structure(df):

    counts = (
        df.groupby(
            [
                "generator_id",
                "timestamp",
            ],
            dropna=False,
        )
        .size()
    )

    duplicate_groups = int(
        (counts > 1).sum()
    )

    duplicate_rows = int(
        (counts[counts > 1] - 1).sum()
    )

    return duplicate_rows, duplicate_groups


def validate_duplicate_structure(source, v3):

    print_section(
        "DUPLICATE GENERATOR/TIMESTAMP VALIDATION"
    )

    (
        source_duplicate_rows,
        source_duplicate_groups,
    ) = duplicate_structure(source)

    (
        v3_duplicate_rows,
        v3_duplicate_groups,
    ) = duplicate_structure(v3)

    print(
        "Source duplicate generator/timestamp rows:",
        source_duplicate_rows,
    )

    print(
        "Source duplicate generator/timestamp groups:",
        source_duplicate_groups,
    )

    print(
        "V3 duplicate generator/timestamp rows:",
        v3_duplicate_rows,
    )

    print(
        "V3 duplicate generator/timestamp groups:",
        v3_duplicate_groups,
    )

    if source_duplicate_rows != v3_duplicate_rows:
        raise AssertionError(
            "Duplicate generator/timestamp row "
            "structure changed."
        )

    if source_duplicate_groups != v3_duplicate_groups:
        raise AssertionError(
            "Duplicate generator/timestamp group "
            "structure changed."
        )

    if source_duplicate_groups > 0:

        print("\nWARNING:")

        print(
            "Duplicate generator/timestamp rows exist "
            "in the source dataset."
        )

        print(
            "They are NOT treated as unique row keys."
        )

    print(
        "[PASS] Duplicate generator/timestamp "
        "structure preserved."
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_identity_columns(df):

    result = df.copy()

    if "timestamp" in result.columns:

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        )

    for column in [
        "generator_id",
        "site_name",
        "status",
    ]:

        if column in result.columns:

            result[column] = (
                result[column]
                .astype("string")
            )

    for column in [
        "fuel_level_l",
        "current",
        "battery_voltage",
        TARGET_COLUMN,
    ]:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


# ============================================================
# CANONICAL ROW REPRESENTATION
# ============================================================

def canonicalize_rows(df):

    result = normalize_identity_columns(
        df[
            TELEMETRY_COLUMNS
            + [TARGET_COLUMN]
        ].copy()
    )

    # --------------------------------------------------------
    # Strings
    # --------------------------------------------------------

    for column in [
        "generator_id",
        "site_name",
        "status",
    ]:

        result[column] = (
            result[column]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
        )

    # --------------------------------------------------------
    # Timestamp
    #
    # Convert to a uniform STRING representation.
    # This avoids mixed-type comparison problems between
    # integer timestamps and "<NA>".
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["timestamp"] = (
        timestamps
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )
        .fillna("<NA>")
    )

    # --------------------------------------------------------
    # Numeric values
    # --------------------------------------------------------

    numeric_columns = [
        "fuel_level_l",
        "current",
        "battery_voltage",
        TARGET_COLUMN,
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        values = values.round(10)

        result[column] = (
            values
            .map(
                lambda x:
                "<NA>"
                if pd.isna(x)
                else f"{float(x):.10f}"
            )
        )

    return result


# ============================================================
# TARGET / TELEMETRY COMPARISON
# ============================================================

def compare_targets(source, v3):

    print_section(
        "1. TARGET INTEGRITY"
    )

    source_target = pd.to_numeric(
        source[TARGET_COLUMN],
        errors="coerce",
    )

    v3_target = pd.to_numeric(
        v3[TARGET_COLUMN],
        errors="coerce",
    )

    source_missing = int(
        source_target.isna().sum()
    )

    v3_missing = int(
        v3_target.isna().sum()
    )

    source_negative = int(
        (source_target < 0).sum()
    )

    v3_negative = int(
        (v3_target < 0).sum()
    )

    print(
        f"Source rows: {len(source):,}"
    )

    print(
        f"V3 rows:     {len(v3):,}"
    )

    print(
        f"\nSource missing targets: {source_missing:,}"
    )

    print(
        f"V3 missing targets:    {v3_missing:,}"
    )

    print(
        f"\nSource negative targets: {source_negative:,}"
    )

    print(
        f"V3 negative targets:    {v3_negative:,}"
    )

    if source_missing != v3_missing:
        raise AssertionError(
            "Target missing-value count changed."
        )

    if source_negative != v3_negative:
        raise AssertionError(
            "Target negative-value count changed."
        )

    print("\nIMPORTANT:")

    print(
        "Row matching is order-independent."
    )

    print(
        "generator_id + timestamp is NOT assumed unique."
    )

    print(
        "Duplicate rows are preserved."
    )

    print(
        "Floating-point serialization is normalized."
    )

    print(
        "No source row order is used."
    )

    # --------------------------------------------------------
    # Canonical multiset comparison
    # --------------------------------------------------------

    source_canonical = canonicalize_rows(
        source
    )

    v3_canonical = canonicalize_rows(
        v3
    )

    source_counts = (
        source_canonical
        .value_counts(
            dropna=False
        )
    )

    v3_counts = (
        v3_canonical
        .value_counts(
            dropna=False
        )
    )

    source_dict = source_counts.to_dict()
    v3_dict = v3_counts.to_dict()

    all_keys = (
        set(source_dict.keys())
        | set(v3_dict.keys())
    )

    source_only_count = 0
    v3_only_count = 0

    for key in all_keys:

        source_count = int(
            source_dict.get(
                key,
                0,
            )
        )

        v3_count = int(
            v3_dict.get(
                key,
                0,
            )
        )

        if source_count > v3_count:

            source_only_count += (
                source_count
                - v3_count
            )

        elif v3_count > source_count:

            v3_only_count += (
                v3_count
                - source_count
            )

    print(
        "\nSource canonical row count:",
        len(source_canonical),
    )

    print(
        "V3 canonical row count:    ",
        len(v3_canonical),
    )

    print(
        "Distinct source canonical rows:",
        len(source_counts),
    )

    print(
        "Distinct V3 canonical rows:    ",
        len(v3_counts),
    )

    print(
        "\nRows present in source but missing from V3:",
        source_only_count,
    )

    print(
        "Rows present in V3 but missing from source:",
        v3_only_count,
    )

    if (
        source_only_count != 0
        or v3_only_count != 0
    ):

        print(
            "\n[FAIL] Source and V3 original rows differ."
        )

        raise AssertionError(
            "Source and V3 original telemetry/target "
            "rows do not match."
        )

    print(
        "\n[PASS] Source and V3 original telemetry/target "
        "rows match."
    )

    # --------------------------------------------------------
    # Independent target comparison
    # --------------------------------------------------------

    source_sorted = (
        source_target
        .sort_values(
            kind="mergesort",
            na_position="first",
        )
        .reset_index(drop=True)
    )

    v3_sorted = (
        v3_target
        .sort_values(
            kind="mergesort",
            na_position="first",
        )
        .reset_index(drop=True)
    )

    both_nan = (
        source_sorted.isna()
        & v3_sorted.isna()
    )

    numeric_equal = (
        source_sorted.notna()
        & v3_sorted.notna()
        & np.isclose(
            source_sorted.fillna(0).to_numpy(
                dtype=float
            ),
            v3_sorted.fillna(0).to_numpy(
                dtype=float
            ),
            rtol=0,
            atol=EPSILON,
        )
    )

    equal = (
        both_nan
        | numeric_equal
    )

    mismatches = int(
        (~equal).sum()
    )

    print(
        "\nTarget mismatches:",
        mismatches,
    )

    if mismatches:
        raise AssertionError(
            "target_fuel_3h values were modified."
        )

    print(
        "[PASS] target_fuel_3h preserved."
    )


# ============================================================
# NEGATIVE TARGET VALIDATION
# ============================================================

def validate_negative_targets(source, v3):

    print_section(
        "2. NEGATIVE TARGET PRESERVATION"
    )

    source_target = pd.to_numeric(
        source[TARGET_COLUMN],
        errors="coerce",
    )

    v3_target = pd.to_numeric(
        v3[TARGET_COLUMN],
        errors="coerce",
    )

    source_negative = source_target[
        source_target < 0
    ]

    v3_negative = v3_target[
        v3_target < 0
    ]

    print(
        "Source negative targets:",
        len(source_negative),
    )

    print(
        "V3 negative targets:",
        len(v3_negative),
    )

    if (
        len(source_negative)
        != len(v3_negative)
    ):

        raise AssertionError(
            "Negative target count changed."
        )

    source_024 = int(
        np.isclose(
            source_negative.to_numpy(
                dtype=float
            ),
            -0.24,
            atol=EPSILON,
            rtol=0,
        ).sum()
    )

    v3_024 = int(
        np.isclose(
            v3_negative.to_numpy(
                dtype=float
            ),
            -0.24,
            atol=EPSILON,
            rtol=0,
        ).sum()
    )

    print(
        "\nSource -0.24 targets:",
        source_024,
    )

    print(
        "V3 -0.24 targets:",
        v3_024,
    )

    if source_024 != v3_024:

        raise AssertionError(
            "-0.24 target count changed."
        )

    print(
        "[PASS] Negative targets preserved."
    )


# ============================================================
# MISSING TARGET VALIDATION
# ============================================================

def validate_missing_targets(source, v3):

    print_section(
        "3. MISSING TARGET PRESERVATION"
    )

    source_missing = int(
        source[TARGET_COLUMN]
        .isna()
        .sum()
    )

    v3_missing = int(
        v3[TARGET_COLUMN]
        .isna()
        .sum()
    )

    print(
        "Source missing targets:",
        source_missing,
    )

    print(
        "V3 missing targets:",
        v3_missing,
    )

    if source_missing != v3_missing:

        raise AssertionError(
            "Missing target count changed."
        )

    print(
        "[PASS] Missing targets preserved."
    )


# ============================================================
# FEATURE LEAKAGE CHECK
# ============================================================

# ============================================================
# FEATURE LEAKAGE CHECK
# ============================================================

def validate_feature_leakage(v3):

    print_section(
        "4. TARGET-DERIVED FEATURE CHECK"
    )

    columns = set(v3.columns)

    # --------------------------------------------------------
    # LEGITIMATE TARGET-ALIGNMENT / DIAGNOSTIC COLUMNS
    # --------------------------------------------------------
    #
    # These columns describe how target_fuel_3h was constructed.
    # They are NOT future telemetry values.
    #
    # IMPORTANT:
    # They are allowed in the V3 dataset for validation,
    # but MUST NOT be used as model input features.
    # --------------------------------------------------------

    present_metadata = sorted(
        ALLOWED_TARGET_METADATA & columns
    )

    print(
        "\nAllowed target-alignment / diagnostic columns:"
    )

    if present_metadata:

        for column in present_metadata:
            print(
                f"  [ALLOWED] {column}"
            )

    else:

        print(
            "  None present."
        )

    # --------------------------------------------------------
    # DETECT FORBIDDEN FUTURE-VALUE COLUMNS
    # --------------------------------------------------------

    suspicious = []

    for column in v3.columns:

        lower = column.lower().strip()

        # ----------------------------------------------------
        # Explicitly allowed target metadata
        # ----------------------------------------------------

        if column in ALLOWED_TARGET_METADATA:
            continue

        # ----------------------------------------------------
        # Exact forbidden future-value names
        # ----------------------------------------------------

        if lower in FORBIDDEN_FUTURE_COLUMNS:

            suspicious.append(column)

            continue

        # ----------------------------------------------------
        # Clearly future-value naming
        # ----------------------------------------------------

        if lower.startswith("future_"):

            suspicious.append(column)

            continue

        if lower.endswith("_future"):

            suspicious.append(column)

            continue

        if "_future_" in lower:

            suspicious.append(column)

            continue

        # ----------------------------------------------------
        # Explicit target-future naming
        # ----------------------------------------------------

        if (
            lower.startswith("target_future")
            or lower.startswith("target_value_future")
            or lower.startswith("target_fuel_value")
        ):

            suspicious.append(column)

            continue

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\nPotential future-value leakage:"
    )

    if suspicious:

        for column in sorted(set(suspicious)):

            print(
                f"  [CHECK] {column}"
            )

        raise AssertionError(
            "Potential future-value feature leakage "
            "detected."
        )

    print(
        "  None detected."
    )

    print(
        "\n[PASS] No obvious future-value "
        "feature leakage detected."
    )

# ============================================================
# REQUIRED V3 FEATURES
# ============================================================

def validate_required_features(v3):

    print_section(
        "5. V3 FEATURE AVAILABILITY"
    )

    expected_features = list(
        REQUIRED_FEATURES
    )

    # Optional running probability features.

    if "running_probability" in v3.columns:

        expected_features.extend([
            "running_probability_mean_5min",
            "running_probability_mean_15min",
            "running_probability_mean_30min",
            "running_probability_mean_60min",
        ])

    # Optional telemetry quality features.

    if "telemetry_quality_score" in v3.columns:

        expected_features.extend([
            "telemetry_quality_mean_5min",
            "telemetry_quality_mean_15min",
            "telemetry_quality_mean_30min",
            "telemetry_quality_mean_60min",
        ])

    missing = [
        column
        for column in expected_features
        if column not in v3.columns
    ]

    if missing:

        print(
            "\nMissing expected V3 features:"
        )

        for column in missing:

            print(
                f"  [MISSING] {column}"
            )

        raise AssertionError(
            "Expected V3 features are missing."
        )

    print(
        f"Expected feature checks: "
        f"{len(expected_features)}"
    )

    print(
        "[PASS] Required V3 features exist."
    )


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

def validate_infinite_values(v3):

    print_section(
        "6. NUMERIC VALUE VALIDATION"
    )

    numeric = v3.select_dtypes(
        include=[np.number]
    )

    if numeric.empty:

        print(
            "No numeric columns found."
        )

        return

    array = numeric.to_numpy(
        dtype=float,
        copy=False,
    )

    infinite_count = int(
        np.isinf(array).sum()
    )

    print(
        "Infinite numeric values:",
        infinite_count,
    )

    if infinite_count:

        raise AssertionError(
            "Infinite numeric values detected."
        )

    print(
        "[PASS] No infinite numeric values."
    )


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

def validate_generator_order(v3):

    print_section(
        "7. CHRONOLOGICAL ORDER VALIDATION"
    )

    ordered = True

    for generator_id, group in v3.groupby(
        "generator_id",
        sort=False,
    ):

        timestamps = pd.to_datetime(
            group["timestamp"],
            errors="coerce",
        )

        valid_timestamps = (
            timestamps.dropna()
        )

        if not valid_timestamps.is_monotonic_increasing:

            print(
                f"[FAIL] {generator_id} "
                "is not chronologically ordered."
            )

            ordered = False

    if not ordered:

        raise AssertionError(
            "V3 is not chronologically ordered "
            "within one or more generators."
        )

    print(
        "[PASS] Chronological order preserved "
        "within each generator."
    )


# ============================================================
# TARGET VALIDITY FLAG
# ============================================================

def validate_target_valid_flag(v3):

    print_section(
        "8. TARGET VALIDITY FLAG"
    )

    if (
        "target_valid_for_regression"
        not in v3.columns
    ):

        raise AssertionError(
            "target_valid_for_regression is missing."
        )

    target = pd.to_numeric(
        v3[TARGET_COLUMN],
        errors="coerce",
    )

    expected = (
        target.notna()
        & (target >= 0)
    ).astype(int)

    actual = (
        pd.to_numeric(
            v3[
                "target_valid_for_regression"
            ],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    mismatches = int(
        (expected != actual).sum()
    )

    print(
        "Flag mismatches:",
        mismatches,
    )

    if mismatches:

        raise AssertionError(
            "target_valid_for_regression "
            "does not match target values."
        )

    print(
        "[PASS] Target validity flag is correct."
    )


# ============================================================
# TARGET AVAILABILITY
# ============================================================

def print_target_availability(v3):

    print_section(
        "9. TARGET AVAILABILITY BY GENERATOR"
    )

    availability = (
        v3.groupby(
            "generator_id"
        )[TARGET_COLUMN]
        .agg(
            records="size",
            valid_targets=lambda x:
                x.notna().sum(),
            negative_targets=lambda x:
                (x < 0).sum(),
        )
    )

    availability[
        "availability_pct"
    ] = (
        availability["valid_targets"]
        / availability["records"]
        * 100
    )

    print(
        availability.round(2)
    )


# ============================================================
# FEATURE NUMERIC SANITY
# ============================================================

def validate_feature_numeric_sanity(v3):

    print_section(
        "10. FEATURE NUMERIC SANITY"
    )

    numeric_columns = (
        v3.select_dtypes(
            include=[np.number]
        )
        .columns
    )

    invalid_summary = []

    for column in numeric_columns:

        series = pd.to_numeric(
            v3[column],
            errors="coerce",
        )

        values = series.to_numpy(
            dtype=float
        )

        inf_count = int(
            np.isinf(values).sum()
        )

        if inf_count:

            invalid_summary.append(
                (
                    column,
                    inf_count,
                )
            )

    if invalid_summary:

        for column, count in invalid_summary:

            print(
                f"  [INVALID] {column}: "
                f"{count:,} infinite values"
            )

        raise AssertionError(
            "One or more V3 features contain "
            "infinite values."
        )

    print(
        "[PASS] Numeric feature values are finite."
    )


# ============================================================
# TARGET DIAGNOSTIC CONSISTENCY
# ============================================================

def validate_target_diagnostics(v3):

    print_section(
        "11. TARGET DIAGNOSTIC CONSISTENCY"
    )

    target = pd.to_numeric(
        v3[TARGET_COLUMN],
        errors="coerce",
    )

    # --------------------------------------------------------
    # target_negative
    # --------------------------------------------------------

    if "target_negative" in v3.columns:

        expected_negative = (
            target.notna()
            & (target < 0)
        ).astype(int)

        actual_negative = (
            pd.to_numeric(
                v3["target_negative"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
        )

        mismatches = int(
            (
                expected_negative
                != actual_negative
            ).sum()
        )

        print(
            "target_negative mismatches:",
            mismatches,
        )

        if mismatches:

            raise AssertionError(
                "target_negative diagnostic "
                "does not match target values."
            )

    # --------------------------------------------------------
    # target_negative_024
    # --------------------------------------------------------

    if "target_negative_024" in v3.columns:

        target_values = target.to_numpy(
            dtype=float
        )

        expected_024 = (
            target.notna()
            & np.isclose(
                target_values,
                -0.24,
                atol=EPSILON,
                rtol=0,
            )
        ).astype(int)

        actual_024 = (
            pd.to_numeric(
                v3["target_negative_024"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
        )

        mismatches = int(
            (
                expected_024
                != actual_024
            ).sum()
        )

        print(
            "target_negative_024 mismatches:",
            mismatches,
        )

        if mismatches:

            raise AssertionError(
                "target_negative_024 diagnostic "
                "does not match target values."
            )

    print(
        "[PASS] Target diagnostic flags are consistent."
    )


# ============================================================
# MODEL FEATURE SAFETY REPORT
# ============================================================

def report_target_derived_columns(v3):

    print_section(
        "12. TARGET-DERIVED COLUMN REPORT"
    )

    target_derived = []

    for column in v3.columns:

        lower = column.lower()

        if (
            column in ALLOWED_TARGET_METADATA
            or lower.startswith("target_")
        ):

            target_derived.append(
                column
            )

    if target_derived:

        print(
            "Target-derived / target-diagnostic columns:"
        )

        for column in sorted(
            set(target_derived)
        ):

            print(
                f"  [TARGET-DERIVED] {column}"
            )

        print(
            "\nIMPORTANT:"
        )

        print(
            "These columns are permitted in the V3 dataset"
        )

        print(
            "for target diagnostics and validation."
        )

        print(
            "They MUST be excluded from model input features."
        )

    else:

        print(
            "No target-derived columns detected."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(source, v3):

    print_header(
        "FINAL V3 VALIDATION SUMMARY"
    )

    print(
        f"Validator version: "
        f"{V3_VALIDATOR_VERSION}"
    )

    print(
        f"\nSource rows: {len(source):,}"
    )

    print(
        f"V3 rows:    {len(v3):,}"
    )

    print(
        f"V3 columns: {len(v3.columns):,}"
    )

    source_target = pd.to_numeric(
        source[TARGET_COLUMN],
        errors="coerce",
    )

    v3_target = pd.to_numeric(
        v3[TARGET_COLUMN],
        errors="coerce",
    )

    print(
        "\nTarget integrity:"
    )

    print(
        "  Source negative targets:",
        int(
            (source_target < 0).sum()
        ),
    )

    print(
        "  V3 negative targets:",
        int(
            (v3_target < 0).sum()
        ),
    )

    print(
        "  Source missing targets:",
        int(
            source_target.isna().sum()
        ),
    )

    print(
        "  V3 missing targets:",
        int(
            v3_target.isna().sum()
        ),
    )

    duplicate_rows, duplicate_groups = (
        duplicate_structure(v3)
    )

    print(
        "\nDuplicate generator/timestamp:"
    )

    print(
        "  Duplicate rows:",
        duplicate_rows,
    )

    print(
        "  Duplicate groups:",
        duplicate_groups,
    )

    print(
        "\nValidation status:"
    )

    print(
        "  ALL CHECKS PASSED"
    )

    print(
        "\nV3 dataset is ready for the "
        "next forecasting stage."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "V3 FORECASTING FEATURE VALIDATION"
    )

    print(
        f"Validator version: "
        f"{V3_VALIDATOR_VERSION}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    source = load_dataset(
        SOURCE_FILE,
        "source dataset",
    )

    v3 = load_dataset(
        V3_FILE,
        "V3 feature dataset",
    )

    # --------------------------------------------------------
    # Basic
    # --------------------------------------------------------

    validate_required_source_columns(
        source,
        v3,
    )

    validate_row_count(
        source,
        v3,
    )

    validate_column_names(
        v3,
    )

    validate_source_columns(
        source,
        v3,
    )

    # --------------------------------------------------------
    # Duplicate structure
    # --------------------------------------------------------

    validate_duplicate_structure(
        source,
        v3,
    )

    # --------------------------------------------------------
    # Target integrity
    # --------------------------------------------------------

    compare_targets(
        source,
        v3,
    )

    validate_negative_targets(
        source,
        v3,
    )

    validate_missing_targets(
        source,
        v3,
    )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    validate_feature_leakage(
        v3
    )

    validate_required_features(
        v3
    )

    validate_infinite_values(
        v3
    )

    validate_generator_order(
        v3
    )

    validate_target_valid_flag(
        v3
    )

    validate_feature_numeric_sanity(
        v3
    )

    validate_target_diagnostics(
        v3
    )

    # --------------------------------------------------------
    # Target-derived reporting
    # --------------------------------------------------------

    report_target_derived_columns(
        v3
    )

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    print_target_availability(
        v3
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        source,
        v3,
    )


if __name__ == "__main__":
    main()