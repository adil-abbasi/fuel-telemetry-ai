from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path("data/processed/forecasting/train_v3.csv")
VALIDATION_PATH = Path("data/processed/forecasting/validation_v3.csv")
TEST_PATH = Path("data/processed/forecasting/test_v3.csv")
FEATURES_PATH = Path("data/processed/forecasting/model_features_v3.txt")

MODEL_DIR = Path("models/fuel_forecasting")
RESULTS_DIR = Path("data/processed/forecasting/results")

MODEL_PATH = MODEL_DIR / "xgboost_baseline.json"
METRICS_PATH = RESULTS_DIR / "xgboost_baseline_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "xgboost_baseline_test_predictions.csv"
IMPORTANCE_PATH = RESULTS_DIR / "xgboost_baseline_feature_importance.csv"


# ============================================================
# UTILITIES
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_feature_list() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found: {FEATURES_PATH}"
        )

    features = [
        line.strip()
        for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not features:
        raise ValueError("Feature list is empty.")

    return features


def load_dataset(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = features + ["target_fuel_3h"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    return df


def prepare_xy(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[features].copy()
    y = pd.to_numeric(
        df["target_fuel_3h"],
        errors="coerce",
    )

    # Convert everything to numeric defensively.
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace infinities.
    X = X.replace([np.inf, -np.inf], np.nan)

    # XGBoost can handle missing values.
    # Do not impute using future information.
    valid_target = y.notna()

    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    return X, y


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE is unstable near zero.
    nonzero = np.abs(y_true) > 1e-6

    if nonzero.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero] - y_pred[nonzero])
                    / y_true[nonzero]
                )
            )
            * 100
        )
    else:
        mape = None

    return {
        "mae_litres": float(mae),
        "rmse_litres": float(rmse),
        "r2": float(r2),
        "mape_percent": (
            float(mape)
            if mape is not None
            else None
        ),
    }


def evaluate(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict, np.ndarray]:

    predictions = model.predict(X)

    metrics = calculate_metrics(
        y.to_numpy(),
        predictions,
    )

    return metrics, predictions


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header("XGBOOST FUEL FORECASTING BASELINE")

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    print("\nLoading feature list...")

    features = load_feature_list()

    print(f"Model features: {len(features)}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_df = load_dataset(
        TRAIN_PATH,
        features,
    )

    validation_df = load_dataset(
        VALIDATION_PATH,
        features,
    )

    test_df = load_dataset(
        TEST_PATH,
        features,
    )

    print(f"Train rows:       {len(train_df):,}")
    print(f"Validation rows:  {len(validation_df):,}")
    print(f"Test rows:        {len(test_df):,}")

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    print_header("PREPARING MODEL DATA")

    X_train, y_train = prepare_xy(
        train_df,
        features,
    )

    X_validation, y_validation = prepare_xy(
        validation_df,
        features,
    )

    X_test, y_test = prepare_xy(
        test_df,
        features,
    )

    print(f"X_train:       {X_train.shape}")
    print(f"X_validation:  {X_validation.shape}")
    print(f"X_test:        {X_test.shape}")

    print(f"y_train:       {len(y_train):,}")
    print(f"y_validation:  {len(y_validation):,}")
    print(f"y_test:        {len(y_test):,}")

    # --------------------------------------------------------
    # FINAL SAFETY CHECKS
    # --------------------------------------------------------

    print_header("MODEL DATA VALIDATION")

    if len(X_train) != len(y_train):
        raise ValueError("Train X/y length mismatch.")

    if len(X_validation) != len(y_validation):
        raise ValueError("Validation X/y length mismatch.")

    if len(X_test) != len(y_test):
        raise ValueError("Test X/y length mismatch.")

    if list(X_train.columns) != features:
        raise ValueError("Training feature order mismatch.")

    if list(X_validation.columns) != features:
        raise ValueError("Validation feature order mismatch.")

    if list(X_test.columns) != features:
        raise ValueError("Test feature order mismatch.")

    non_numeric = [
        column
        for column in X_train.columns
        if not pd.api.types.is_numeric_dtype(
            X_train[column]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric features detected: {non_numeric}"
        )

    print("[PASS] Train X/y aligned.")
    print("[PASS] Validation X/y aligned.")
    print("[PASS] Test X/y aligned.")
    print("[PASS] Feature ordering consistent.")
    print("[PASS] All model features numeric.")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print_header("TRAINING XGBOOST BASELINE")

    model = XGBRegressor(
        objective="reg:squarederror",

        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,

        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.0,
        reg_lambda=1.0,

        random_state=42,
        n_jobs=-1,

        tree_method="hist",
    )

    print("Model configuration:")
    print(f"  n_estimators:     {model.n_estimators}")
    print(f"  learning_rate:    {model.learning_rate}")
    print(f"  max_depth:        {model.max_depth}")
    print(f"  min_child_weight: {model.min_child_weight}")
    print(f"  subsample:        {model.subsample}")
    print(f"  colsample_bytree: {model.colsample_bytree}")

    print("\nStarting training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_validation, y_validation),
        ],
        verbose=False,
    )

    print("[PASS] XGBoost training completed.")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print_header("VALIDATION RESULTS")

    validation_metrics, validation_predictions = evaluate(
        model,
        X_validation,
        y_validation,
    )

    print(
        f"MAE:   {validation_metrics['mae_litres']:.4f} L"
    )

    print(
        f"RMSE:  {validation_metrics['rmse_litres']:.4f} L"
    )

    print(
        f"R²:    {validation_metrics['r2']:.6f}"
    )

    if validation_metrics["mape_percent"] is not None:
        print(
            f"MAPE:  {validation_metrics['mape_percent']:.2f}%"
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print_header("FINAL TEST RESULTS")

    test_metrics, test_predictions = evaluate(
        model,
        X_test,
        y_test,
    )

    print(
        f"MAE:   {test_metrics['mae_litres']:.4f} L"
    )

    print(
        f"RMSE:  {test_metrics['rmse_litres']:.4f} L"
    )

    print(
        f"R²:    {test_metrics['r2']:.6f}"
    )

    if test_metrics["mape_percent"] is not None:
        print(
            f"MAPE:  {test_metrics['mape_percent']:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print_header("SAVING BASELINE")

    model.save_model(MODEL_PATH)

    print(f"Model saved:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_output = {
        "model": "XGBoost baseline",
        "target": "target_fuel_3h",

        "features": len(features),

        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),

        "validation": validation_metrics,
        "test": test_metrics,

        "configuration": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Metrics saved:")
    print(METRICS_PATH)

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_output = test_df.loc[
        y_test.index,
        [
            column
            for column in [
                "generator_id",
                "site_name",
                "timestamp",
                "fuel_level_l",
                "target_fuel_3h",
            ]
            if column in test_df.columns
        ]
    ].copy()

    prediction_output["prediction_fuel_3h"] = test_predictions

    prediction_output["prediction_error_l"] = (
        prediction_output["prediction_fuel_3h"]
        - prediction_output["target_fuel_3h"]
    )

    prediction_output["absolute_error_l"] = (
        prediction_output["prediction_error_l"]
        .abs()
    )

    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Test predictions saved:")
    print(PREDICTIONS_PATH)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print("Feature importance saved:")
    print(IMPORTANCE_PATH)

    print("\nTop 20 features:")

    for _, row in importance.head(20).iterrows():
        print(
            f"{row['feature']:<45} "
            f"{row['importance']:.6f}"
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print_header("XGBOOST BASELINE COMPLETE")

    print("Target:")
    print("  target_fuel_3h")

    print("\nValidation:")
    print(
        f"  MAE:  {validation_metrics['mae_litres']:.4f} L"
    )
    print(
        f"  RMSE: {validation_metrics['rmse_litres']:.4f} L"
    )
    print(
        f"  R²:   {validation_metrics['r2']:.6f}"
    )

    print("\nTest:")
    print(
        f"  MAE:  {test_metrics['mae_litres']:.4f} L"
    )
    print(
        f"  RMSE: {test_metrics['rmse_litres']:.4f} L"
    )
    print(
        f"  R²:   {test_metrics['r2']:.6f}"
    )

    print("\nNext stage:")
    print("  Analyze baseline errors and feature importance.")
    print("  Then tune the model using validation data only.")


if __name__ == "__main__":
    main()