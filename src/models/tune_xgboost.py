from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data/processed/forecasting/train_v3.csv"
VALIDATION_PATH = BASE_DIR / "data/processed/forecasting/validation_v3.csv"
TEST_PATH = BASE_DIR / "data/processed/forecasting/test_v3.csv"
FEATURES_PATH = BASE_DIR / "data/processed/forecasting/model_features_v3.txt"

MODEL_DIR = BASE_DIR / "models/fuel_forecasting"
RESULT_DIR = BASE_DIR / "data/processed/forecasting/results/xgboost_tuning"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "target_fuel_3h"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_data():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train rows:       {len(train):,}")
    print(f"Validation rows:  {len(validation):,}")
    print(f"Test rows:        {len(test):,}")

    return train, validation, test


def prepare_xy(df, features):
    X = df[features].copy()
    y = df[TARGET].astype(float).copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # XGBoost handles missing numeric values.
    X = X.astype(np.float32)

    return X, y


def print_results(name, results):
    print()
    print(name)
    print("-" * 50)
    print(f"MAE:  {results['mae']:.4f} L")
    print(f"RMSE: {results['rmse']:.4f} L")
    print(f"R²:   {results['r2']:.6f}")


def main():

    print("=" * 70)
    print("XGBOOST FUEL FORECASTING - VALIDATION TUNING")
    print("=" * 70)

    features = load_features()

    print(f"\nModel features: {len(features)}")

    train, validation, test = load_data()

    X_train, y_train = prepare_xy(train, features)
    X_val, y_val = prepare_xy(validation, features)
    X_test, y_test = prepare_xy(test, features)

    print("\nPreparing model matrices...")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_val.shape}")
    print(f"X_test:       {X_test.shape}")

    # ---------------------------------------------------------
    # BASELINE CONFIGURATION
    # ---------------------------------------------------------

    baseline_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # ---------------------------------------------------------
    # TUNING CONFIGURATIONS
    #
    # Test ONLY against validation.
    # Test dataset is NEVER used here.
    # ---------------------------------------------------------

    configs = [

        {
            "name": "baseline",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        {
            "name": "shallower",
            "n_estimators": 700,
            "learning_rate": 0.04,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },

        {
            "name": "regularized",
            "n_estimators": 700,
            "learning_rate": 0.04,
            "max_depth": 7,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },

        {
            "name": "strong_regularization",
            "n_estimators": 800,
            "learning_rate": 0.035,
            "max_depth": 6,
            "min_child_weight": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        {
            "name": "deeper",
            "n_estimators": 600,
            "learning_rate": 0.045,
            "max_depth": 9,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

    results = []
    models = {}

    print("\n" + "=" * 70)
    print("VALIDATION TUNING")
    print("=" * 70)

    for config in configs:

        name = config["name"]

        print("\n" + "-" * 70)
        print(f"Training configuration: {name}")
        print("-" * 70)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,

            "n_estimators": config["n_estimators"],
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "min_child_weight": config["min_child_weight"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
        }

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)

        metrics = evaluate(y_val, pred)

        row = {
            "name": name,
            **config,
            **metrics,
        }

        results.append(row)
        models[name] = model

        print_results(f"Validation: {name}", metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("VALIDATION TUNING RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "name",
                "mae",
                "rmse",
                "r2",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
            ]
        ].to_string(index=False)
    )

    results_path = RESULT_DIR / "validation_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    best_name = results_df.iloc[0]["name"]
    best_model = models[best_name]

    print("\n" + "=" * 70)
    print("BEST VALIDATION MODEL")
    print("=" * 70)

    print(f"Model: {best_name}")

    best_row = results_df.iloc[0]

    print(f"Validation MAE:  {best_row['mae']:.4f} L")
    print(f"Validation RMSE: {best_row['rmse']:.4f} L")
    print(f"Validation R²:   {best_row['r2']:.6f}")

    # ---------------------------------------------------------
    # SAVE BEST MODEL
    # ---------------------------------------------------------

    best_model_path = MODEL_DIR / "xgboost_tuned_validation_best.json"

    best_model.save_model(best_model_path)

    print("\nBest model saved:")
    print(best_model_path)

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    importance = best_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importance,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
        RESULT_DIR / "xgboost_tuned_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\nTop 20 tuned features:")

    print(
        importance_df.head(20).to_string(index=False)
    )

    # ---------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # ---------------------------------------------------------

    val_predictions = validation[
        [
            c for c in
            ["generator_id", "site_name", "timestamp",
             "fuel_level_l", TARGET]
            if c in validation.columns
        ]
    ].copy()

    val_predictions["prediction_fuel_3h"] = models[
        best_name
    ].predict(X_val)

    val_predictions["prediction_error_l"] = (
        val_predictions["prediction_fuel_3h"]
        - val_predictions[TARGET]
    )

    val_predictions["absolute_error_l"] = (
        val_predictions["prediction_error_l"].abs()
    )

    val_prediction_path = (
        RESULT_DIR / "best_model_validation_predictions.csv"
    )

    val_predictions.to_csv(
        val_prediction_path,
        index=False,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # DO NOT EVALUATE TEST HERE.
    #
    # Test remains untouched until the tuned model has been
    # selected.
    # ---------------------------------------------------------

    metadata = {
        "selected_model": best_name,
        "selection_metric": "validation_mae",
        "test_used_for_selection": False,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(features),
        "target": TARGET,
        "baseline_validation": {
            "mae": float(
                results_df.loc[
                    results_df["name"] == "baseline",
                    "mae"
                ].iloc[0]
            ),
            "rmse": float(
                results_df.loc[
                    results_df["name"] == "baseline",
                    "rmse"
                ].iloc[0]
            ),
            "r2": float(
                results_df.loc[
                    results_df["name"] == "baseline",
                    "r2"
                ].iloc[0]
            ),
        },
        "best_validation": {
            "mae": float(best_row["mae"]),
            "rmse": float(best_row["rmse"]),
            "r2": float(best_row["r2"]),
        },
    }

    metadata_path = RESULT_DIR / "tuning_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("VALIDATION TUNING COMPLETE")
    print("=" * 70)

    print(f"Best model: {best_name}")
    print(f"Validation MAE:  {best_row['mae']:.4f} L")
    print(f"Validation RMSE: {best_row['rmse']:.4f} L")
    print(f"Validation R²:   {best_row['r2']:.6f}")

    print("\nSaved:")
    print(results_path)
    print(best_model_path)
    print(importance_path)
    print(val_prediction_path)
    print(metadata_path)

    print("\nTEST SET WAS NOT USED FOR MODEL SELECTION.")


if __name__ == "__main__":
    main()