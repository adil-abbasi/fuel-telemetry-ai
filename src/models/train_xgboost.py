from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ======================================================
# PATHS
# ======================================================

TRAIN_PATH = Path(
    "data/processed/train_fuel_forecasting.csv"
)

VALIDATION_PATH = Path(
    "data/processed/validation_fuel_forecasting.csv"
)

TEST_PATH = Path(
    "data/processed/test_fuel_forecasting.csv"
)

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / (
    "fuel_xgboost_v1.json"
)

IMPORTANCE_PATH = MODEL_DIR / (
    "fuel_xgboost_v1_feature_importance.csv"
)


# ======================================================
# FEATURES
# ======================================================

FEATURE_COLUMNS = [
    "fuel_level_l",
    "fuel_delta",
    "current",
    "current_delta",
    "battery_voltage",
    "voltage_delta",
    "running_probability",
    "hour",
    "minute",
    "weekday",
    "is_weekend",
    "time_delta_sec",
    "telemetry_quality_score",
]

TARGET_COLUMN = "target_fuel_3h"


# ======================================================
# CONFIGURATION
# ======================================================

RANDOM_STATE = 42

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ======================================================
# LOAD DATA
# ======================================================

def load_dataset(path):

    print(
        f"Loading: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    return df


# ======================================================
# PREPARE DATA
# ======================================================

def prepare_dataset(df):

    required = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    # Ensure numeric values.
    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    # Valid target only.
    valid = (
        y.notna()
        &
        (y >= 0)
    )

    X = X.loc[valid]
    y = y.loc[valid]

    return X, y


# ======================================================
# METRICS
# ======================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return mae, rmse, r2


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("XGBOOST FUEL FORECASTING V1")
    print("=" * 70)

    # ==================================================
    # LOAD
    # ==================================================

    train_df = load_dataset(
        TRAIN_PATH
    )

    validation_df = load_dataset(
        VALIDATION_PATH
    )

    test_df = load_dataset(
        TEST_PATH
    )

    print("\nDataset sizes")

    print(
        f"Training:    {len(train_df):,}"
    )

    print(
        f"Validation:  {len(validation_df):,}"
    )

    print(
        f"Testing:     {len(test_df):,}"
    )

    # ==================================================
    # PREPARE
    # ==================================================

    print("\nPreparing features...")

    X_train, y_train = prepare_dataset(
        train_df
    )

    X_validation, y_validation = (
        prepare_dataset(
            validation_df
        )
    )

    X_test, y_test = prepare_dataset(
        test_df
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation):,}"
    )

    print(
        f"Test samples: "
        f"{len(X_test):,}"
    )

    print(
        f"\nFeatures: "
        f"{len(FEATURE_COLUMNS)}"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"  - {feature}"
        )

    # ==================================================
    # TRAIN
    # ==================================================

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model = xgb.XGBRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_train,
                y_train,
            ),
            (
                X_validation,
                y_validation,
            ),
        ],
        verbose=100,
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    validation_pred = model.predict(
        X_validation
    )

    val_mae, val_rmse, val_r2 = (
        calculate_metrics(
            y_validation,
            validation_pred,
        )
    )

    print(
        f"\nMAE:  {val_mae:.4f} L"
    )

    print(
        f"RMSE: {val_rmse:.4f} L"
    )

    print(
        f"R²:   {val_r2:.6f}"
    )

    # ==================================================
    # TEST
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    test_pred = model.predict(
        X_test
    )

    test_mae, test_rmse, test_r2 = (
        calculate_metrics(
            y_test,
            test_pred,
        )
    )

    print(
        f"\nMAE:  {test_mae:.4f} L"
    )

    print(
        f"RMSE: {test_rmse:.4f} L"
    )

    print(
        f"R²:   {test_r2:.6f}"
    )

    # ==================================================
    # BASELINE COMPARISON
    # ==================================================

    baseline_valid = (
        test_df["fuel_level_l"].notna()
        &
        test_df[TARGET_COLUMN].notna()
    )

    baseline_y_true = test_df.loc[
        baseline_valid,
        TARGET_COLUMN,
    ]

    baseline_y_pred = test_df.loc[
        baseline_valid,
        "fuel_level_l",
    ]

    baseline_mae, baseline_rmse, baseline_r2 = (
        calculate_metrics(
            baseline_y_true,
            baseline_y_pred,
        )
    )

    print("\n" + "=" * 70)
    print("BASELINE VS XGBOOST")
    print("=" * 70)

    print(
        f"\n{'Metric':<12}"
        f"{'Baseline':>15}"
        f"{'XGBoost':>15}"
    )

    print(
        f"{'MAE':<12}"
        f"{baseline_mae:>15.4f}"
        f"{test_mae:>15.4f}"
    )

    print(
        f"{'RMSE':<12}"
        f"{baseline_rmse:>15.4f}"
        f"{test_rmse:>15.4f}"
    )

    print(
        f"{'R²':<12}"
        f"{baseline_r2:>15.6f}"
        f"{test_r2:>15.6f}"
    )

    # ==================================================
    # IMPROVEMENT
    # ==================================================

    if baseline_mae != 0:

        mae_improvement = (
            (
                baseline_mae
                - test_mae
            )
            / baseline_mae
            * 100
        )

        print(
            f"\nMAE improvement: "
            f"{mae_improvement:.2f}%"
        )

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    # ==================================================
    # SAVE TEST PREDICTIONS
    # ==================================================

    predictions = test_df[
        [
            "generator_id",
            "timestamp",
            "fuel_level_l",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions["xgboost_prediction"] = (
        test_pred
    )

    predictions["xgboost_error"] = (
        predictions[TARGET_COLUMN]
        - predictions["xgboost_prediction"]
    )

    predictions["absolute_error"] = (
        predictions["xgboost_error"]
        .abs()
    )

    predictions_path = MODEL_DIR / (
        "fuel_xgboost_v1_test_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print("\nSaved model:")

    print(
        MODEL_PATH
    )

    print("\nSaved feature importance:")

    print(
        IMPORTANCE_PATH
    )

    print("\nSaved test predictions:")

    print(
        predictions_path
    )

    print("\n" + "=" * 70)
    print("XGBOOST V1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()