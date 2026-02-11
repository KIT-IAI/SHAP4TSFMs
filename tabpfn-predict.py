import json
import os
import platform
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn_time_series import (
    FeatureTransformer,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tqdm import tqdm
from workalendar.europe import BadenWurttemberg

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

TRANSNET_FEATURES = [
    ("temperature", "data/Air_Temperature_2m.csv"),
    ("irradiance", "data/Global_Horizontal_Irradiance.csv"),
]


# Set device based on operating system
if platform.system() == "Linux":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif platform.system() == "Darwin":  # macOS
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


def get_transnetbw_df():
    END_DATE = "2025-09-30 23:59:00"
    transnet_load_df = pd.read_csv(
        "data/TransnetBW_Total_Load.csv", parse_dates=["Timestamp"]
    )
    transnet_load_df = pd.DataFrame(
        {
            "id": ["transnet"] * (len(transnet_load_df) // 4),
            "timestamp": transnet_load_df["Timestamp"][::4],
            "target": np.asarray(transnet_load_df["Actual_Total_Load"])
            .reshape((-1, 4))
            .mean(axis=-1),
        }
    )
    calendar = BadenWurttemberg()
    transnet_load_df["holiday"] = [
        1 if (timepoint.dayofweek == 6 or calendar.is_holiday(timepoint)) else 0
        for timepoint in transnet_load_df["timestamp"]
    ]
    transnet_load_df = transnet_load_df[transnet_load_df["timestamp"] <= END_DATE]
    for feature, file in TRANSNET_FEATURES:
        feature_df = pd.read_csv(
            file, parse_dates=["Timestamp"], index_col="Timestamp"
        )[:END_DATE]
        transnet_load_df[feature] = np.asarray(feature_df)[::4, :4].mean(axis=-1)
    return transnet_load_df


class TabPFNEnsemble:
    def __init__(self, **tabpfn_kwargs):
        self.model_z = TabPFNRegressor(**tabpfn_kwargs)
        self.model_power = TabPFNRegressor(**tabpfn_kwargs)
        self.scaler_z = StandardScaler()
        self.scaler_power = PowerTransformer(method="box-cox", standardize=True)
        self.use_power = True
        self.shift = 0

    def fit(self, X_train, y_train):
        # Extract values for target transformation
        y_train_values = y_train.values if hasattr(y_train, "values") else y_train
        y_train_values = y_train_values.reshape(-1, 1)

        # Z-normalized model - TabPFN takes DataFrame for X, array for y
        y_z = self.scaler_z.fit_transform(y_train_values)
        self.model_z.fit(X_train, y_z.ravel())

        # Power-transformed model
        try:
            y_train_shifted = y_train_values.copy()
            if y_train_values.min() <= 0:
                self.shift = abs(y_train_values.min()) + 1e-6
                y_train_shifted = y_train_values + self.shift
            else:
                self.shift = 0
            y_power = self.scaler_power.fit_transform(y_train_shifted)
            self.model_power.fit(X_train, y_power.ravel())
        except Exception as e:
            print(f"Power transform failed: {e}. Using z-normalization only.")
            self.use_power = False
        return self

    def predict(self, X_test):
        # TabPFN predict expects DataFrame, not array
        y_pred_z = self.scaler_z.inverse_transform(
            self.model_z.predict(X_test, output_type="median").reshape(-1, 1)
        ).ravel()

        if not self.use_power:
            return y_pred_z

        y_pred_power = (
            self.scaler_power.inverse_transform(
                self.model_power.predict(X_test, output_type="median").reshape(-1, 1)
            ).ravel()
            - self.shift
        )

        return (y_pred_z + y_pred_power) / 2


def main(context_name: str, context_length: int):
    # Generate unique run ID to save results
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stride = 1
    os.makedirs("results_tabpfn", exist_ok=True)
    START_DATE = datetime(2024, 10, 1)
    END_DATE = datetime(2025, 9, 30)

    target = "target"
    prediction_length = 24
    id_column = "id"
    timestamp_column = "timestamp"

    # Save configuration
    config = {
        "run_id": run_id,
        "context_name": context_name,
        "context_length": context_length,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "stride": stride,
        "prediction_length": prediction_length,
        "device": DEVICE,
    }
    with open(f"results_tabpfn/{run_id}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    transnet_df = get_transnetbw_df()
    print(transnet_df)

    # Calculate number of predictions based on stride
    total_hours = ((END_DATE - START_DATE).days + 1) * 24 - prediction_length + 1
    n_predictions = total_hours // stride
    print(f"Total predictions to make: {n_predictions}\n")

    print(f"\n{'=' * 80}")
    print(f"Testing Context Length: {context_name} ({context_length} timesteps)")
    print(f"{'=' * 80}\n")

    all_predictions = []
    all_ground_truths = []

    # Create feature transformer once (reuse across all predictions)
    feature_transformer = FeatureTransformer(
        [
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ]
    )

    # Initialize ensemble model once and reuse
    print("Initializing ensemble model (will be reused across predictions)...")
    pipeline = TabPFNEnsemble(
        device=DEVICE,
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=-1,
    )

    # Process predictions sequentially
    print(f"Processing {n_predictions} predictions...")

    for pred_i in tqdm(range(n_predictions), desc=f"{context_name}"):
        start_date = START_DATE + timedelta(hours=pred_i * stride)
        energy_context_df = transnet_df[transnet_df[timestamp_column] < start_date]
        # Check if we have enough data
        if len(energy_context_df) < context_length:
            if pred_i == 0:
                print(
                    f"WARNING: Not enough data for context length {context_length}. Available: {len(energy_context_df)}"
                )
                print(f"Skipping {context_name}...\n")
            break

        energy_context_df = energy_context_df[-context_length:]
        energy_future_df = transnet_df[transnet_df[timestamp_column] >= start_date]
        energy_future_df = energy_future_df[:prediction_length]

        if len(energy_future_df) < prediction_length:
            break

        ground_truth = np.array(energy_future_df[target])

        # Convert to TimeSeriesDataFrame format
        train_tsdf = TimeSeriesDataFrame(
            energy_context_df.rename(columns={id_column: "item_id"}).set_index(
                ["item_id", "timestamp"]
            )
        )

        test_tsdf = TimeSeriesDataFrame(
            energy_future_df.rename(columns={id_column: "item_id"}).set_index(
                ["item_id", "timestamp"]
            )
        )
        test_tsdf_copy = test_tsdf.copy()
        test_tsdf_copy["target"] = np.nan

        # Transform features
        train_tsdf_tf, test_tsdf_tf = feature_transformer.transform(
            train_tsdf, test_tsdf_copy
        )
        unneeded_features = [
            "second_of_minute_sin",
            "second_of_minute_cos",
            "minute_of_hour_sin",
            "minute_of_hour_cos",
        ]  # Unneeded since we have only hourly data
        train_tsdf_tf.drop(columns=unneeded_features, inplace=True)
        test_tsdf_tf.drop(columns=unneeded_features, inplace=True)

        if pred_i == 0:
            print(f"\nFeature columns: {train_tsdf_tf.columns.tolist()}")

        # Extract features and target in TabPFN API format
        X_tr, y_tr = (
            train_tsdf_tf.drop(columns=target),
            train_tsdf_tf[target],
        )
        X_te = test_tsdf_tf.drop(columns=target)

        # Fit and predict using ensemble
        pipeline.fit(X_tr, y_tr)
        predictions = pipeline.predict(X_te)

        all_predictions.append(predictions)
        all_ground_truths.append(ground_truth)

    assert len(all_predictions)

    all_predictions = np.array(all_predictions)
    all_ground_truths = np.array(all_ground_truths)
    print("Shape of predictions and ground truths:")
    print(all_predictions.shape)
    print(all_ground_truths.shape)

    # Evaluate
    mae = np.mean(np.abs(all_predictions - all_ground_truths))
    mse = np.mean(np.square(all_predictions - all_ground_truths))
    rmse = np.sqrt(mse)
    mape = (
        np.mean(np.abs((all_predictions - all_ground_truths) / all_ground_truths)) * 100
    )

    print(
        f"{context_name}: {len(all_predictions)} predictions | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%"
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY OF RESULTS - TABPFN-TS")
    print(f"{'=' * 80}")
    print(f"Start Date: {START_DATE}")
    print(f"End Date: {END_DATE}")
    print(f"Prediction Length: {prediction_length} hours")
    print(f"Stride: {stride} hour(s)")
    print(f"{'=' * 80}\n")

    print(
        f"{'Context':<15} {'Length':<10} {'N_Pred':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}"
    )
    print("-" * 80)
    print(
        f"{context_name:<15} "
        f"{context_length:<10} "
        f"{len(all_predictions):<10} "
        f"{mae:<12.2f} "
        f"{rmse:<12.2f} "
        f"{mape:<12.2f}%"
    )
    print("=" * 80)

    # Create results DataFrame
    results_df = pd.DataFrame(
        [
            {
                "Context": context_name,
                "Context_Length": context_length,
                "N_Predictions": len(all_predictions),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        ]
    )

    # Save results to CSV with run_id
    results_df.to_csv(
        f"results_tabpfn/stride{stride}/{run_id}_results.csv", index=False
    )
    print(f"\nResults saved to 'results_tabpfn/stride{stride}/{run_id}_results.csv'")

    print(f"\nAll results saved with ID: {run_id}")


if __name__ == "__main__":
    import typer

    typer.run(main)
