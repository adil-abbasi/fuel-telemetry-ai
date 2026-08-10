"""
MODEL-SAFE FORECASTING DATA PREPARATION

Purpose
-------
Prepare V3 feature-engineered data for forecasting models.

Pipeline
--------
1. Load V3 dataset.
2. Sort chronologically within each generator.
3. Identify target-derived columns.
4. Remove target-derived columns from model inputs.
5. Keep target_fuel_3h as the prediction target.
6. Mark invalid targets:
       - missing target
       - negative target
7. Split chronologically per generator:
       70% train
       15% validation
       15% test
8. Save train/validation/test datasets.
9. Print detailed diagnostics.

Important
---------
This script does NOT randomly shuffle the data.

Target-derived columns are never allowed into X.

Target:
    target_fuel_3h

Invalid targets:
    NaN
    target < 0

The source/V3 dataset may contain duplicate
(generator_id, timestamp) rows.

generator_id + timestamp is therefore NOT treated
as a unique key.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

V3_FILE = Path(
    "data/processed/fuel_forecasting_features_v3.csv"
)

OUTPUT_DIR = Path(
    "data/processed/forecasting"
)

TARGET_COLUMN = "target_fuel_3h"


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# TARGET-DERIVED COLUMNS
# ============================================================

TARGET_DERIVED_COLUMNS = {
    # Actual target
    "target_fuel_3h",

    # Target alignment metadata
    "target_timestamp",
    "target_lookup_time",
    "target_time_difference_minutes",

    # Target diagnostics
    "target_negative",
    "target_negative_024",
    "target_valid_for_regression",
}


# ============================================================
# ID / METADATA COLUMNS
# ============================================================

IDENTITY_COLUMNS = {
    "timestamp",
    "generator_id",
    "site_name",
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_v3():

    print_header(
        "LOADING V3 FORECASTING DATASET"
    )

    if not V3_FILE.exists():

        raise FileNotFoundError(
            f"V3 dataset not found:\n{V3_FILE}"
        )

    df = pd.read_csv(
        V3_FILE,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_input(df):

    print_section(
        "INPUT VALIDATION"
    )

    required = {
        "timestamp",
        "generator_id",
        TARGET_COLUMN,
    }

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "[PASS] Required columns exist."
    )

    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Duplicate column names found:\n"
            + str(duplicate_columns)
        )

    print(
        "[PASS] Column names are unique."
    )


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

def sort_data(df):

    print_section(
        "CHRONOLOGICAL SORT"
    )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    missing_timestamp = int(
        result["timestamp"].isna().sum()
    )

    print(
        "Missing timestamps:",
        missing_timestamp,
    )

    if missing_timestamp:

        print(
            "WARNING: rows with missing timestamps "
            "will be excluded from model splitting."
        )

    result = result.sort_values(
        [
            "generator_id",
            "timestamp",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(
        drop=True
    )

    print(
        "[PASS] Data sorted chronologically "
        "within each generator."
    )

    return result


# ============================================================
# TARGET VALIDITY
# ============================================================

def analyze_target(df):

    print_section(
        "TARGET ANALYSIS"
    )

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    missing = int(
        target.isna().sum()
    )

    negative = int(
        (target < 0).sum()
    )

    valid = int(
        (
            target.notna()
            & (target >= 0)
        ).sum()
    )

    print(
        f"Total rows:       {len(df):,}"
    )

    print(
        f"Valid targets:     {valid:,}"
    )

    print(
        f"Missing targets:   {missing:,}"
    )

    print(
        f"Negative targets:  {negative:,}"
    )

    print(
        f"Valid percentage:  "
        f"{valid / len(df) * 100:.2f}%"
    )

    return target


# ============================================================
# REMOVE ROWS THAT CANNOT BE USED FOR SUPERVISED LEARNING
# ============================================================

def remove_invalid_targets(df):

    print_section(
        "TARGET VALIDITY FILTER"
    )

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    valid_mask = (
        target.notna()
        & (target >= 0)
        & df["timestamp"].notna()
        & df["generator_id"].notna()
    )

    valid_df = df.loc[
        valid_mask
    ].copy()

    removed = len(df) - len(valid_df)

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Usable rows:    {len(valid_df):,}"
    )

    print(
        f"Removed rows:   {removed:,}"
    )

    print(
        "\nRows removed because they have:"
    )

    print(
        f"  Missing target: "
        f"{int(target.isna().sum()):,}"
    )

    print(
        f"  Negative target: "
        f"{int((target < 0).sum()):,}"
    )

    print(
        f"  Missing timestamp: "
        f"{int(df['timestamp'].isna().sum()):,}"
    )

    print(
        "[PASS] Only valid supervised-learning "
        "targets retained."
    )

    return valid_df


# ============================================================
# MODEL FEATURE IDENTIFICATION
# ============================================================

def identify_model_features(df):

    print_section(
        "MODEL FEATURE SELECTION"
    )

    excluded = set(
        TARGET_DERIVED_COLUMNS
    )

    feature_columns = []

    for column in df.columns:

        if column in excluded:
            continue

        if column in IDENTITY_COLUMNS:
            continue

        feature_columns.append(
            column
        )

    print(
        f"Total V3 columns:     {len(df.columns)}"
    )

    print(
        f"Model input features:  {len(feature_columns)}"
    )

    print(
        "\nExcluded identity columns:"
    )

    for column in sorted(
        IDENTITY_COLUMNS
        & set(df.columns)
    ):

        print(
            f"  [IDENTITY] {column}"
        )

    print(
        "\nExcluded target-derived columns:"
    )

    present_target_columns = sorted(
        TARGET_DERIVED_COLUMNS
        & set(df.columns)
    )

    for column in present_target_columns:

        print(
            f"  [TARGET] {column}"
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    leaked = (
        set(feature_columns)
        & TARGET_DERIVED_COLUMNS
    )

    if leaked:

        raise AssertionError(
            "TARGET LEAKAGE DETECTED.\n"
            f"These target-derived columns would enter X:\n"
            f"{sorted(leaked)}"
        )

    print(
        "\n[PASS] No target-derived columns "
        "will be used as model inputs."
    )

    print(
        "\nModel features:"
    )

    for index, column in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"  {index:03d}. {column}"
        )

    return feature_columns


# ============================================================
# NUMERIC FEATURE CHECK
# ============================================================

# ============================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================

# ============================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================

CATEGORICAL_FEATURES = [
    "site_quality",
    "status",
    "estimated_status",
    "confidence_level",
    "estimation_reason",
    "sensor_votes",
    "validation_reason",
    "imputation_confidence_level",
]

# Maximum number of categories allowed for one-hot encoding.
# Anything above this is frequency encoded.
ONE_HOT_MAX_CATEGORIES = 20


def encode_categorical_features(df, feature_columns):
    """
    Encode categorical model features while preserving all identity /
    temporal columns required later for chronological splitting.

    Important:
    - generator_id must remain in df
    - timestamp must remain in df
    - site_name must remain in df
    - only feature_columns are transformed
    """

    print("\nCategorical features detected:")

    categorical_columns = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):

            categorical_columns.append(column)

            print(
                f"[CATEGORICAL] {column}: "
                f"{df[column].nunique(dropna=False):,} unique values"
            )

    if not categorical_columns:

        print("No categorical features detected.")

        return df, feature_columns

    encoded_parts = []

    remaining_feature_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    # --------------------------------------------------------
    # Process categorical columns individually
    # --------------------------------------------------------

    for column in categorical_columns:

        series = df[column].astype("string").fillna("<MISSING>")

        unique_count = series.nunique(dropna=False)

        print(f"\nProcessing: {column}")

        # ----------------------------------------------------
        # High-cardinality categorical feature
        # ----------------------------------------------------

        if unique_count > 50:

            print("Encoding method: FREQUENCY")
            print(f"Categories: {unique_count:,}")

            frequencies = series.value_counts(
                normalize=True,
                dropna=False,
            )

            encoded = (
                series
                .map(frequencies)
                .astype(np.float32)
            )

            encoded_name = f"{column}__frequency"

            encoded_parts.append(
                pd.DataFrame(
                    {
                        encoded_name: encoded
                    },
                    index=df.index,
                )
            )

            print(
                f"Created column: {encoded_name}"
            )

        # ----------------------------------------------------
        # Low-cardinality categorical feature
        # ----------------------------------------------------

        else:

            print("Encoding method: ONE-HOT")
            print(f"Categories: {unique_count}")

            encoded = pd.get_dummies(
                series,
                prefix=column,
                prefix_sep="__",
                dtype=np.int8,
            )

            encoded.index = df.index

            encoded_parts.append(encoded)

            print(
                f"Created columns: {encoded.shape[1]}"
            )

    # --------------------------------------------------------
    # Build encoded feature dataframe
    # --------------------------------------------------------

    numeric_features = df[
        remaining_feature_columns
    ].copy()

    # Convert numeric model features explicitly.

    for column in numeric_features.columns:

        numeric_features[column] = pd.to_numeric(
            numeric_features[column],
            errors="coerce",
        ).astype(np.float32)

    # --------------------------------------------------------
    # Combine numeric + encoded features
    # --------------------------------------------------------

    encoded_features = pd.concat(
        [
            numeric_features
        ] + encoded_parts,
        axis=1,
    )

    # --------------------------------------------------------
    # Preserve identity / temporal columns
    # --------------------------------------------------------

    identity_columns = []

    for column in [
        "generator_id",
        "site_name",
        "timestamp",
        "target_fuel_3h",
    ]:

        if column in df.columns:

            identity_columns.append(column)

    # Preserve any other non-model columns already required
    # by downstream validation/splitting.

    preserved_columns = df[
        identity_columns
    ].copy()

    # --------------------------------------------------------
    # Combine preserved columns + model features
    # --------------------------------------------------------

    result = pd.concat(
        [
            preserved_columns,
            encoded_features,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    final_feature_columns = list(
        encoded_features.columns
    )

    print(
        f"\nOriginal model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Categorical features processed: "
        f"{len(categorical_columns)}"
    )

    print(
        f"Final numeric model features: "
        f"{len(final_feature_columns)}"
    )

    print(
        f"Final dataframe columns: "
        f"{len(result.columns)}"
    )

    print(
        f"Final rows: "
        f"{len(result):,}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "\n[PASS] Categorical features encoded."
    )

    if not all(
        pd.api.types.is_numeric_dtype(
            result[column]
        )
        for column in final_feature_columns
    ):

        raise ValueError(
            "Some encoded model features are still non-numeric."
        )

    print(
        "[PASS] All model features are numeric."
    )

    numeric = result[
        final_feature_columns
    ].select_dtypes(
        include=[np.number]
    )

    infinite_count = int(
        np.isinf(
            numeric.to_numpy(
                dtype=np.float64
            )
        ).sum()
    )

    print(
        f"[PASS] No infinite values created."
        if infinite_count == 0
        else f"[FAIL] Infinite values: {infinite_count}"
    )

    if infinite_count:

        raise ValueError(
            "Infinite values detected after categorical encoding."
        )

    return result, final_feature_columns
# ============================================================
# MODEL FEATURE VALIDATION
# ============================================================

def validate_model_features(df, feature_columns):

    print("\n" + "=" * 70)
    print("MODEL FEATURE VALIDATION")
    print("=" * 70)

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:

        print("\nMissing model features:")

        for column in missing_features:
            print(
                f"[MISSING] {column}"
            )

        raise ValueError(
            "Required model features are missing."
        )

    # --------------------------------------------------------
    # Check numeric
    # --------------------------------------------------------

    non_numeric = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            non_numeric.append(
                column
            )

    if non_numeric:

        print(
            "\nNon-numeric model features:"
        )

        for column in non_numeric:

            print(
                f"[NON-NUMERIC] {column}"
            )

        raise ValueError(
            "Non-numeric columns detected in model features."
        )

    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    numeric_data = df[
        feature_columns
    ]

    values = numeric_data.to_numpy(
        dtype=float
    )

    infinite_count = int(
        np.isinf(values).sum()
    )

    if infinite_count:

        raise ValueError(
            f"{infinite_count:,} infinite values "
            "detected in model features."
        )

    # --------------------------------------------------------
    # Check duplicate feature names
    # --------------------------------------------------------

    if len(feature_columns) != len(
        set(feature_columns)
    ):

        raise ValueError(
            "Duplicate model feature names detected."
        )

    print(
        f"\nModel features validated: "
        f"{len(feature_columns)}"
    )

    print(
        "[PASS] All model features are numeric."
    )

    print(
        "[PASS] No infinite model feature values."
    )

    print(
        "[PASS] Model feature names are unique."
    )

# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def chronological_split(
    df,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
):
    """
    Chronological train/validation/test split.

    Split is performed independently for each generator.

    IMPORTANT:
    generator_id and timestamp are metadata columns and are NOT
    model features, but they must remain available here.
    """

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    required_columns = [
        "generator_id",
        "timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Chronological split requires columns: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Validate ratios
    # --------------------------------------------------------

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if not np.isclose(
        total_ratio,
        1.0,
    ):

        raise ValueError(
            "train_ratio + validation_ratio + "
            "test_ratio must equal 1.0"
        )

    # --------------------------------------------------------
    # Sort globally by generator + timestamp
    # --------------------------------------------------------

    working = df.copy()

    working["timestamp"] = pd.to_datetime(
        working["timestamp"],
        errors="coerce",
    )

    working = (
        working
        .sort_values(
            [
                "generator_id",
                "timestamp",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    train_parts = []
    validation_parts = []
    test_parts = []

    # --------------------------------------------------------
    # Split independently per generator
    # --------------------------------------------------------

    for generator_id, group in working.groupby(
        "generator_id",
        sort=False,
    ):

        group = group.reset_index(
            drop=True
        )

        n = len(group)

        if n < 3:

            raise ValueError(
                f"Generator {generator_id} has only "
                f"{n} rows. Cannot perform chronological split."
            )

        train_end = int(
            n * train_ratio
        )

        validation_end = train_end + int(
            n * validation_ratio
        )

        # Guarantee non-empty partitions.

        train_end = max(
            1,
            min(
                train_end,
                n - 2,
            ),
        )

        validation_end = max(
            train_end + 1,
            min(
                validation_end,
                n - 1,
            ),
        )

        train_group = group.iloc[
            :train_end
        ].copy()

        validation_group = group.iloc[
            train_end:validation_end
        ].copy()

        test_group = group.iloc[
            validation_end:
        ].copy()

        train_parts.append(
            train_group
        )

        validation_parts.append(
            validation_group
        )

        test_parts.append(
            test_group
        )

        print(
            f"{generator_id}: "
            f"total={n:,} | "
            f"train={len(train_group):,} | "
            f"validation={len(validation_group):,} | "
            f"test={len(test_group):,}"
        )

    # --------------------------------------------------------
    # Combine generators
    # --------------------------------------------------------

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort each split chronologically
    # --------------------------------------------------------

    for split in [
        train_df,
        validation_df,
        test_df,
    ]:

        split.sort_values(
            [
                "generator_id",
                "timestamp",
            ],
            kind="mergesort",
            inplace=True,
        )

        split.reset_index(
            drop=True,
            inplace=True,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(working):

        raise ValueError(
            "Chronological split lost or duplicated rows."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "CHRONOLOGICAL SPLIT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total rows:       {len(working):,}"
    )

    print(
        f"Training rows:    {len(train_df):,}"
    )

    print(
        f"Validation rows:  {len(validation_df):,}"
    )

    print(
        f"Testing rows:     {len(test_df):,}"
    )

    print(
        f"Split total:      {total_split_rows:,}"
    )

    print(
        "\n[PASS] Chronological split preserved all rows."
    )

    return (
        train_df,
        validation_df,
        test_df,
    )

# ============================================================
# SPLIT ORDER VALIDATION
# ============================================================

def validate_split_order(
    train_df,
    validation_df,
    test_df,
):

    print_section(
        "SPLIT ORDER VALIDATION"
    )

    for generator_id in sorted(
        set(train_df["generator_id"])
        | set(validation_df["generator_id"])
        | set(test_df["generator_id"])
    ):

        train = train_df[
            train_df["generator_id"]
            == generator_id
        ]

        validation = validation_df[
            validation_df["generator_id"]
            == generator_id
        ]

        test = test_df[
            test_df["generator_id"]
            == generator_id
        ]

        if (
            len(train)
            and len(validation)
        ):

            train_end = train[
                "timestamp"
            ].max()

            validation_start = validation[
                "timestamp"
            ].min()

            if validation_start < train_end:

                raise AssertionError(
                    f"Chronological leakage detected "
                    f"between train and validation for "
                    f"{generator_id}."
                )

        if (
            len(validation)
            and len(test)
        ):

            validation_end = validation[
                "timestamp"
            ].max()

            test_start = test[
                "timestamp"
            ].min()

            if test_start < validation_end:

                raise AssertionError(
                    f"Chronological leakage detected "
                    f"between validation and test for "
                    f"{generator_id}."
                )

    print(
        "[PASS] Train/validation/test chronological "
        "ordering is valid."
    )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    train_df,
    validation_df,
    test_df,
    feature_columns,
):

    print_header(
        "SAVING MODEL DATASETS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save complete split datasets
    #
    # These retain target + metadata for evaluation.
    # --------------------------------------------------------

    train_file = (
        OUTPUT_DIR
        / "train_v3.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / "validation_v3.csv"
    )

    test_file = (
        OUTPUT_DIR
        / "test_v3.csv"
    )

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_file = (
        OUTPUT_DIR
        / "model_features_v3.txt"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8",
    ) as file:

        for column in feature_columns:

            file.write(
                f"{column}\n"
            )

    print(
        f"Train dataset:\n{train_file}"
    )

    print(
        f"Validation dataset:\n{validation_file}"
    )

    print(
        f"Test dataset:\n{test_file}"
    )

    print(
        f"Feature list:\n{feature_file}"
    )

    return (
        train_file,
        validation_file,
        test_file,
        feature_file,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    train_df,
    validation_df,
    test_df,
    feature_columns,
):

    print_header(
        "MODEL DATA PREPARATION COMPLETE"
    )

    print(
        f"Model features: {len(feature_columns)}"
    )

    print(
        f"\nTrain rows:      {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Test rows:       {len(test_df):,}"
    )

    print(
        f"\nTotal rows:      "
        f"{len(train_df) + len(validation_df) + len(test_df):,}"
    )

    print(
        "\nTarget:"
    )

    print(
        f"  {TARGET_COLUMN}"
    )

    print(
        "\nTarget-derived columns excluded from X:"
    )

    for column in sorted(
        TARGET_DERIVED_COLUMNS
    ):

        print(
            f"  {column}"
        )

    print(
        "\nSplit strategy:"
    )

    print(
        "  70% chronological training"
    )

    print(
        "  15% chronological validation"
    )

    print(
        "  15% chronological testing"
    )

    print(
        "\nShuffle:"
    )

    print(
        "  DISABLED"
    )

    print(
        "\n[PASS] Model-ready forecasting datasets created."
    )

    print(
        "\nNext stage:"
    )

    print(
        "  Train XGBoost baseline."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "V3 MODEL DATA PREPARATION"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_v3()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(
        df
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = sort_data(
        df
    )

    # --------------------------------------------------------
    # Target analysis
    # --------------------------------------------------------

    analyze_target(
        df
    )

    # --------------------------------------------------------
    # Remove invalid supervised targets
    # --------------------------------------------------------

    df = remove_invalid_targets(
        df
    )

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    feature_columns = identify_model_features(
        df
    )

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    # ============================================================
    # ENCODE CATEGORICAL FEATURES
    # ============================================================

    df, feature_columns = encode_categorical_features(
        df,
        feature_columns,
    )

    # ============================================================
    # VALIDATE FINAL MODEL FEATURES
    # ============================================================

    validate_model_features(
        df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
    train_df,
    validation_df,
    test_df,
) = chronological_split(
    df
)

    # --------------------------------------------------------
    # Validate chronological boundaries
    # --------------------------------------------------------

    validate_split_order(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_datasets(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()