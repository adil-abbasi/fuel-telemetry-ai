from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = (
    BASE_DIR
    / "data/processed/forecasting/test_v3.csv"
)

# Full V3 history is required for a proper persistence baseline.
V3_PATH = (
    BASE_DIR
    / "data/processed/fuel_forecasting_features_v3.csv"
)

FEATURES_PATH = (
    BASE_DIR
    / "data/processed/forecasting/model_features_v3.txt"
)

TUNED_MODEL_PATH = (
    BASE_DIR
    / "models/fuel_forecasting/"
    "xgboost_tuned_validation_best.json"
)

BASELINE_MODEL_PATH = (
    BASE_DIR
    / "models/fuel_forecasting/"
    "xgboost_baseline.json"
)

RESULT_DIR = (
    BASE_DIR
    / "data/processed/forecasting/results/"
    "xgboost_tuning"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET = "target_fuel_3h"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction):
    """
    Calculate regression metrics.

    Both arrays must contain only finite values.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if len(y_true) != len(prediction):
        raise ValueError(
            "Metric arrays have different lengths: "
            f"y_true={len(y_true)}, "
            f"prediction={len(prediction)}"
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics on an empty dataset."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "prediction contains NaN or infinite values."
        )

    error = prediction - y_true

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                prediction,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    prediction,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                prediction,
            )
        ),
        "mean_error": float(
            np.mean(error)
        ),
        "median_absolute_error": float(
            np.median(np.abs(error))
        ),
        "max_absolute_error": float(
            np.max(np.abs(error))
        ),
        "rows": int(
            len(y_true)
        ),
    }


def print_metrics(name, metrics):
    """Print formatted regression metrics."""

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Rows:                   "
        f"{metrics['rows']:,}"
    )

    print(
        f"MAE:                    "
        f"{metrics['mae']:.4f} L"
    )

    print(
        f"RMSE:                   "
        f"{metrics['rmse']:.4f} L"
    )

    print(
        f"R²:                     "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"Mean error:             "
        f"{metrics['mean_error']:.4f} L"
    )

    print(
        f"Median absolute error:  "
        f"{metrics['median_absolute_error']:.4f} L"
    )

    print(
        f"Maximum absolute error: "
        f"{metrics['max_absolute_error']:.4f} L"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_model_features(test, features):
    """Validate that every model feature exists exactly once."""

    if len(features) != len(set(features)):
        raise ValueError(
            "Duplicate feature names detected in feature list."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in test.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features in test dataset:\n"
            + "\n".join(missing_features)
        )


def validate_prediction(
    name,
    prediction,
    expected_rows,
):
    """Validate prediction shape and numerical validity."""

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if len(prediction) != expected_rows:
        raise ValueError(
            f"{name} prediction length mismatch: "
            f"expected {expected_rows}, "
            f"got {len(prediction)}"
        )

    if not np.isfinite(prediction).all():

        invalid_count = int(
            (~np.isfinite(prediction)).sum()
        )

        raise ValueError(
            f"{name} contains "
            f"{invalid_count} NaN/infinite predictions."
        )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

def build_persistence_baseline(
    test,
    v3_path,
):
    """
    Build a leakage-safe persistence baseline.

    For every test row, prediction is the latest known
    fuel_level_l for the same generator at or before
    the test timestamp.

    Rules:

    1. Same generator only.
    2. Historical timestamp <= test timestamp.
    3. Exact timestamp is allowed.
    4. Future observations are never used.
    5. Invalid historical fuel values are ignored.
    6. Duplicate generator/timestamp observations are
       reduced deterministically.
    7. Original test row order is preserved.

    Returns
    -------
    persistence_prediction : np.ndarray
        Persistence prediction for every test row.
        Missing values remain NaN.

    persistence_valid_mask : np.ndarray
        True where a valid persistence prediction exists.
    """

    print()
    print("=" * 70)
    print("BUILDING PERSISTENCE BASELINE")
    print("=" * 70)

    required_columns = [
        "generator_id",
        "timestamp",
        "fuel_level_l",
    ]

    # ========================================================
    # VALIDATE TEST DATA
    # ========================================================

    missing_test_columns = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing_test_columns:
        raise ValueError(
            "Test dataset is missing required columns: "
            f"{missing_test_columns}"
        )

    test_lookup = test[
        required_columns
    ].copy()

    test_lookup["timestamp"] = pd.to_datetime(
        test_lookup["timestamp"],
        errors="coerce",
    )

    test_lookup["fuel_level_l"] = pd.to_numeric(
        test_lookup["fuel_level_l"],
        errors="coerce",
    )

    if test_lookup["timestamp"].isna().any():
        invalid_count = int(
            test_lookup["timestamp"].isna().sum()
        )

        raise ValueError(
            f"Test dataset contains "
            f"{invalid_count:,} invalid timestamps."
        )

    # Preserve original test row order.
    test_lookup["_test_row_id"] = np.arange(
        len(test_lookup),
        dtype=np.int64,
    )

    # ========================================================
    # LOAD HISTORICAL DATA
    # ========================================================

    if not v3_path.exists():
        raise FileNotFoundError(
            f"Historical V3 dataset not found:\n{v3_path}"
        )

    print()
    print(
        f"Loading historical V3 data:\n{v3_path}"
    )

    historical = pd.read_csv(
        v3_path,
        usecols=required_columns,
    )

    print(
        f"Historical rows: "
        f"{len(historical):,}"
    )

    # ========================================================
    # NORMALIZE HISTORICAL DATA
    # ========================================================

    historical["timestamp"] = pd.to_datetime(
        historical["timestamp"],
        errors="coerce",
    )

    historical["fuel_level_l"] = pd.to_numeric(
        historical["fuel_level_l"],
        errors="coerce",
    )

    # Remove invalid lookup keys.
    historical = historical.dropna(
        subset=[
            "generator_id",
            "timestamp",
        ]
    ).copy()

    # Only actual fuel observations can be used.
    historical = historical[
        historical["fuel_level_l"].notna()
    ].copy()

    # Negative fuel values are invalid.
    historical = historical[
        historical["fuel_level_l"] >= 0
    ].copy()

    # ========================================================
    # HANDLE DUPLICATES
    # ========================================================

    duplicate_mask = historical.duplicated(
        subset=[
            "generator_id",
            "timestamp",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print(
        f"Historical duplicate rows used "
        f"for lookup handling: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        # Deterministic duplicate handling.
        historical = (
            historical
            .sort_values(
                [
                    "timestamp",
                    "generator_id",
                    "fuel_level_l",
                ],
                kind="mergesort",
            )
            .drop_duplicates(
                subset=[
                    "generator_id",
                    "timestamp",
                ],
                keep="last",
            )
            .copy()
        )

    # ========================================================
    # PREPARE AS-OF LOOKUP
    # ========================================================

    test_lookup = test_lookup[
        [
            "_test_row_id",
            "generator_id",
            "timestamp",
        ]
    ].copy()

    historical_lookup = historical[
        [
            "generator_id",
            "timestamp",
            "fuel_level_l",
        ]
    ].copy()

    # IMPORTANT:
    #
    # pandas merge_asof requires the `on` timestamp
    # column to be globally sorted.
    #
    # Therefore timestamp comes FIRST in sorting.
    #

    test_lookup = (
        test_lookup
        .sort_values(
            [
                "timestamp",
                "generator_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    historical_lookup = (
        historical_lookup
        .sort_values(
            [
                "timestamp",
                "generator_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # AS-OF MERGE
    # ========================================================

    persistence = pd.merge_asof(
        test_lookup,
        historical_lookup,
        on="timestamp",
        by="generator_id",
        direction="backward",
        allow_exact_matches=True,
    )

    # ========================================================
    # RESTORE ORIGINAL TEST ORDER
    # ========================================================

    persistence = (
        persistence
        .sort_values(
            "_test_row_id"
        )
        .reset_index(
            drop=True
        )
    )

    persistence_prediction = (
        persistence[
            "fuel_level_l"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # BUILD CORRECT VALID MASK
    # ========================================================

    missing_mask = (
        ~np.isfinite(
            persistence_prediction
        )
    )

    # THIS IS THE IMPORTANT FIX.
    #
    # valid_mask must mean:
    #
    #     True  = persistence prediction exists
    #     False = persistence prediction unavailable
    #
    persistence_valid_mask = (
        ~missing_mask
    )

    available_count = int(
        persistence_valid_mask.sum()
    )

    missing_count = int(
        missing_mask.sum()
    )

    print()
    print(
        f"Persistence predictions available: "
        f"{available_count:,}/{len(test):,}"
    )

    print(
        f"Persistence predictions missing: "
        f"{missing_count:,}"
    )

    # ========================================================
    # REPORT MISSING ROWS
    # ========================================================

    if missing_count > 0:

        print()
        print(
            f"WARNING: {missing_count:,} test rows "
            "have no previous valid fuel observation."
        )

        missing_generators = (
            test.loc[
                missing_mask,
                "generator_id",
            ]
            .value_counts()
            .to_dict()
        )

        print(
            "Missing persistence observations "
            "by generator:"
        )

        for generator, count in (
            missing_generators.items()
        ):
            print(
                f"  {generator}: {count}"
            )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if available_count + missing_count != len(test):
        raise RuntimeError(
            "Persistence availability accounting error: "
            f"available={available_count}, "
            f"missing={missing_count}, "
            f"test_rows={len(test)}"
        )

    return (
        persistence_prediction,
        persistence_valid_mask,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL TEST EVALUATION - TUNED XGBOOST")
    print("=" * 70)

    # ========================================================
    # LOAD FEATURE LIST
    # ========================================================

    print("\nLoading feature list...")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found:\n{FEATURES_PATH}"
        )

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        f"Model features: "
        f"{len(features)}"
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("\nLoading test dataset...")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    test = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    required_columns = [
        "generator_id",
        "timestamp",
        TARGET,
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required test columns:\n"
            + "\n".join(missing_required)
        )

    validate_model_features(
        test,
        features,
    )

    # ========================================================
    # PREPARE MODEL MATRICES
    # ========================================================

    print("\nPreparing model matrices...")

    X_test = test[
        features
    ].copy()

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test = X_test.astype(
        np.float32
    )

    y_test = (
        pd.to_numeric(
            test[TARGET],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(y_test).all():

        invalid_target_count = int(
            (~np.isfinite(y_test)).sum()
        )

        raise ValueError(
            f"Test target contains "
            f"{invalid_target_count:,} "
            f"NaN/infinite values."
        )

    print(
        f"X_test: "
        f"{X_test.shape}"
    )

    print(
        f"y_test: "
        f"{len(y_test):,}"
    )

    # ========================================================
    # LOAD TUNED MODEL
    # ========================================================

    print("\nLoading tuned model...")

    if not TUNED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tuned model not found:\n"
            f"{TUNED_MODEL_PATH}"
        )

    tuned_model = xgb.XGBRegressor()

    tuned_model.load_model(
        TUNED_MODEL_PATH
    )

    print(
        "Tuned model loaded."
    )

    # ========================================================
    # LOAD BASELINE MODEL
    # ========================================================

    print("\nLoading original baseline...")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found:\n"
            f"{BASELINE_MODEL_PATH}"
        )

    baseline_model = xgb.XGBRegressor()

    baseline_model.load_model(
        BASELINE_MODEL_PATH
    )

    print(
        "Baseline model loaded."
    )

    # ========================================================
    # TUNED PREDICTIONS
    # ========================================================

    print(
        "\nGenerating tuned predictions..."
    )

    tuned_prediction = (
        tuned_model.predict(
            X_test
        )
    )

    validate_prediction(
        "Tuned XGBoost",
        tuned_prediction,
        len(test),
    )

    tuned_metrics = calculate_metrics(
        y_test,
        tuned_prediction,
    )

    # ========================================================
    # BASELINE PREDICTIONS
    # ========================================================

    print(
        "Generating baseline predictions..."
    )

    baseline_prediction = (
        baseline_model.predict(
            X_test
        )
    )

    validate_prediction(
        "Original XGBoost",
        baseline_prediction,
        len(test),
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_prediction,
    )

    # ========================================================
    # PERSISTENCE BASELINE
    # ========================================================

    print(
        "\nGenerating persistence baseline..."
    )

    (
        persistence_prediction,
        persistence_valid_mask,
    ) = build_persistence_baseline(
        test,
        V3_PATH,
    )

    # ========================================================
    # PERSISTENCE METRICS
    # ========================================================

    valid_persistence_rows = int(
        persistence_valid_mask.sum()
    )

    print()
    print(
        f"Valid persistence rows: "
        f"{valid_persistence_rows:,}"
    )

    if valid_persistence_rows == 0:
        raise ValueError(
            "Persistence baseline has zero "
            "valid predictions. "
            "Check historical V3 data."
        )

    persistence_metrics = calculate_metrics(
        y_test[
            persistence_valid_mask
        ],
        persistence_prediction[
            persistence_valid_mask
        ],
    )

    # ========================================================
    # COMPARABLE SUBSET
    # ========================================================

    comparable_y = (
        y_test[
            persistence_valid_mask
        ]
    )

    comparable_tuned = (
        tuned_prediction[
            persistence_valid_mask
        ]
    )

    comparable_baseline = (
        baseline_prediction[
            persistence_valid_mask
        ]
    )

    tuned_comparable_metrics = (
        calculate_metrics(
            comparable_y,
            comparable_tuned,
        )
    )

    baseline_comparable_metrics = (
        calculate_metrics(
            comparable_y,
            comparable_baseline,
        )
    )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print_metrics(
        "TUNED XGBOOST - FINAL TEST",
        tuned_metrics,
    )

    print_metrics(
        "ORIGINAL XGBOOST - FINAL TEST",
        baseline_metrics,
    )

    print_metrics(
        "PERSISTENCE BASELINE - VALID ROWS",
        persistence_metrics,
    )

    # ========================================================
    # COMPARABLE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "COMPARABLE TEST SUBSET "
        "(PERSISTENCE AVAILABLE)"
    )
    print("=" * 70)

    print(
        f"Comparable rows: "
        f"{valid_persistence_rows:,}"
    )

    print()

    print(
        f"Tuned XGBoost MAE:     "
        f"{tuned_comparable_metrics['mae']:.4f} L"
    )

    print(
        f"Original XGBoost MAE:  "
        f"{baseline_comparable_metrics['mae']:.4f} L"
    )

    print(
        f"Persistence MAE:       "
        f"{persistence_metrics['mae']:.4f} L"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    baseline_mae = baseline_metrics[
        "mae"
    ]

    tuned_mae = tuned_metrics[
        "mae"
    ]

    persistence_mae = persistence_metrics[
        "mae"
    ]

    if baseline_mae != 0:

        mae_improvement = (
            (
                baseline_mae
                - tuned_mae
            )
            / baseline_mae
            * 100
        )

    else:

        mae_improvement = np.nan

    comparable_tuned_mae = (
        tuned_comparable_metrics[
            "mae"
        ]
    )

    if persistence_mae != 0:

        persistence_improvement = (
            (
                persistence_mae
                - comparable_tuned_mae
            )
            / persistence_mae
            * 100
        )

    else:

        persistence_improvement = np.nan

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Baseline MAE:          "
        f"{baseline_mae:.4f} L"
    )

    print(
        f"Tuned MAE:             "
        f"{tuned_mae:.4f} L"
    )

    print(
        f"Tuning improvement:    "
        f"{mae_improvement:.2f}%"
    )

    print()

    print(
        f"Persistence MAE:       "
        f"{persistence_mae:.4f} L"
    )

    print(
        f"Tuned vs persistence:  "
        f"{persistence_improvement:.2f}%"
    )

    # ========================================================
    # BUILD PREDICTION DATASET
    # ========================================================

    prediction_columns = [
        column
        for column in [
            "generator_id",
            "site_name",
            "timestamp",
            "fuel_level_l",
            TARGET,
        ]
        if column in test.columns
    ]

    predictions = test[
        prediction_columns
    ].copy()

    predictions[
        "tuned_prediction"
    ] = tuned_prediction

    predictions[
        "baseline_prediction"
    ] = baseline_prediction

    predictions[
        "persistence_prediction"
    ] = persistence_prediction

    predictions[
        "persistence_available"
    ] = persistence_valid_mask

    # ========================================================
    # ERRORS
    # ========================================================

    predictions[
        "tuned_error"
    ] = (
        tuned_prediction
        - y_test
    )

    predictions[
        "tuned_absolute_error"
    ] = np.abs(
        predictions[
            "tuned_error"
        ]
    )

    predictions[
        "baseline_error"
    ] = (
        baseline_prediction
        - y_test
    )

    predictions[
        "baseline_absolute_error"
    ] = np.abs(
        predictions[
            "baseline_error"
        ]
    )

    predictions[
        "persistence_error"
    ] = np.where(
        persistence_valid_mask,
        persistence_prediction - y_test,
        np.nan,
    )

    predictions[
        "persistence_absolute_error"
    ] = np.abs(
        predictions[
            "persistence_error"
        ]
    )

    # ========================================================
    # ERROR THRESHOLDS
    # ========================================================

    print()
    print("=" * 70)
    print("TUNED XGBOOST ERROR THRESHOLDS")
    print("=" * 70)

    for threshold in [
        5,
        10,
        20,
        30,
        50,
        100,
        200,
    ]:

        count = int(
            (
                predictions[
                    "tuned_absolute_error"
                ]
                > threshold
            ).sum()
        )

        percentage = (
            count
            / len(predictions)
            * 100
        )

        print(
            f"Tuned error > {threshold:3d} L:"
            f" {count:6,}"
            f" ({percentage:6.2f}%)"
        )

    # ========================================================
    # GENERATOR-LEVEL PERFORMANCE
    # ========================================================

    generator_results = []

    for generator_id, group in (
        predictions.groupby(
            "generator_id",
            sort=False,
        )
    ):

        y = group[
            TARGET
        ].to_numpy(
            dtype=np.float64
        )

        pred = group[
            "tuned_prediction"
        ].to_numpy(
            dtype=np.float64
        )

        metrics = calculate_metrics(
            y,
            pred,
        )

        # ----------------------------------------------------
        # Persistence generator metrics
        # ----------------------------------------------------

        persistence_group = group[
            group[
                "persistence_available"
            ]
        ]

        if len(persistence_group) > 0:

            persistence_generator_metrics = (
                calculate_metrics(
                    persistence_group[
                        TARGET
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    persistence_group[
                        "persistence_prediction"
                    ].to_numpy(
                        dtype=np.float64
                    ),
                )
            )

            persistence_mae_generator = (
                persistence_generator_metrics[
                    "mae"
                ]
            )

        else:

            persistence_mae_generator = np.nan

        generator_results.append({
            "generator_id":
                generator_id,

            "rows":
                int(len(group)),

            "mae":
                metrics["mae"],

            "median_absolute_error":
                metrics[
                    "median_absolute_error"
                ],

            "rmse":
                metrics["rmse"],

            "r2":
                metrics["r2"],

            "mean_error":
                metrics["mean_error"],

            "max_absolute_error":
                metrics[
                    "max_absolute_error"
                ],

            "persistence_mae":
                persistence_mae_generator,

            "persistence_rows":
                int(
                    len(
                        persistence_group
                    )
                ),
        })

    generator_results = (
        pd.DataFrame(
            generator_results
        )
        .sort_values(
            "mae",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 70)
    print("GENERATOR-LEVEL TEST PERFORMANCE")
    print("=" * 70)

    print(
        generator_results.to_string(
            index=False
        )
    )

    # ========================================================
    # WORST PREDICTIONS
    # ========================================================

    worst_predictions = (
        predictions
        .sort_values(
            "tuned_absolute_error",
            ascending=False,
        )
        .head(50)
        .copy()
    )

    print()
    print("=" * 70)
    print("TOP 20 WORST TUNED PREDICTIONS")
    print("=" * 70)

    display_columns = [
        column
        for column in [
            "generator_id",
            "timestamp",
            "fuel_level_l",
            TARGET,
            "tuned_prediction",
            "tuned_error",
            "tuned_absolute_error",
        ]
        if column in worst_predictions.columns
    ]

    print(
        worst_predictions[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_output = {
        "tuned_xgboost":
            tuned_metrics,

        "original_xgboost":
            baseline_metrics,

        "persistence":
            persistence_metrics,

        "comparable_subset": {
            "rows":
                int(
                    persistence_valid_mask.sum()
                ),

            "tuned_xgboost":
                tuned_comparable_metrics,

            "original_xgboost":
                baseline_comparable_metrics,
        },

        "persistence_availability": {
            "total_test_rows":
                int(len(test)),

            "available_rows":
                int(
                    persistence_valid_mask.sum()
                ),

            "unavailable_rows":
                int(
                    (~persistence_valid_mask).sum()
                ),

            "availability_percent":
                float(
                    persistence_valid_mask.mean()
                    * 100
                ),
        },

        "improvement": {
            "tuned_vs_original_mae_percent":
                float(
                    mae_improvement
                ),

            "tuned_vs_persistence_mae_percent":
                float(
                    persistence_improvement
                ),
        },

        "test_rows":
            int(len(test)),

        "feature_count":
            int(len(features)),

        "target":
            TARGET,

        "evaluation_policy": {
            "test_set_used_only_for_final_evaluation":
                True,

            "persistence_uses_future_data":
                False,

            "persistence_direction":
                "backward",

            "persistence_allows_exact_timestamp":
                True,

            "persistence_future_backfill":
                False,
        },
    }

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics_path = (
        RESULT_DIR
        / "final_test_comparison.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_path = (
        RESULT_DIR
        / "final_test_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    # ========================================================
    # SAVE GENERATOR RESULTS
    # ========================================================

    generator_path = (
        RESULT_DIR
        / "final_generator_performance.csv"
    )

    generator_results.to_csv(
        generator_path,
        index=False,
    )

    # ========================================================
    # SAVE WORST PREDICTIONS
    # ========================================================

    worst_path = (
        RESULT_DIR
        / "final_worst_predictions.csv"
    )

    worst_predictions.to_csv(
        worst_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")

    print(
        metrics_path
    )

    print(
        prediction_path
    )

    print(
        generator_path
    )

    print(
        worst_path
    )

    print()
    print("=" * 70)
    print("EVALUATION POLICY")
    print("=" * 70)

    print(
        "Test set was used only for final evaluation."
    )

    print(
        "Persistence uses only historical "
        "fuel observations at or before each "
        "prediction timestamp."
    )

    print(
        "No future observations were used "
        "for persistence."
    )

    unavailable_count = int(
        (~persistence_valid_mask).sum()
    )

    print(
        f"Persistence unavailable for "
        f"{unavailable_count:,} "
        f"test rows."
    )

    print(
        "Those rows remain in the XGBoost "
        "evaluation but are excluded from "
        "the persistence metric."
    )


if __name__ == "__main__":
    main()