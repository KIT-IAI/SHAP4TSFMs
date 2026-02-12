import os
import pickle
import platform
import time
from datetime import datetime, timedelta
from math import factorial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
import torch
import typer
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
    ("temperature", PROJECT_ROOT / "data/Air_Temperature_2m.csv"),
    ("irradiance", PROJECT_ROOT / "data/Global_Horizontal_Irradiance.csv"),
]

WINDOW_CONFIGS = {
    168 * 8: {
        "lengths": [168 * 4, 168 * 3, 24 * 6, 24],
        "names": ["week-8--5", "week-4--2", "day-7--2", "day-1"],
    },
    168 * 48: {
        "lengths": [168 * 24, 168 * 12, 168 * 8, 168 * 3, 24 * 6, 24],
        "names": [
            "week-48--25",
            "week-24--13",
            "week-12--5",
            "week-4--2",
            "day-7--2",
            "day-1",
        ],
    },
    8760: {
        "lengths": [168 * 48 + 24, 168 * 3, 24 * 6, 24],
        "names": ["week-52--5", "week-4--2", "day-7--2", "day-1"],
    },
}
DEFAULT_WINDOW_CONFIG_KEY = 8760

UNNEEDED_HOURLY_FEATURES = [
    "second_of_minute_sin",
    "second_of_minute_cos",
    "minute_of_hour_sin",
    "minute_of_hour_cos",
]


# Set device based on operating system
if platform.system() == "Linux":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif platform.system() == "Darwin":  # macOS
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


def powerset(s):
    """Generate all subsets of a set."""
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def get_transnetbw_df():
    END_DATE = "2025-09-30 23:59:00"
    transnet_load_df = pd.read_csv(
        PROJECT_ROOT / "data/TransnetBW_Total_Load.csv", parse_dates=["Timestamp"]
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
    """Ensemble of TabPFN models with z-normalization and power transformation."""

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

        # Z-normalized model
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


def main(
    context_length: int = 8760,
    start_date: str = "2024-10-01",
):
    """Run TabPFN time series forecasting with subseries explanations."""
    CONTEXT_LENGTH = context_length
    START_DATE = datetime.strptime(start_date, "%Y-%m-%d")
    DAYS = 365

    # Configure window lengths based on context length
    if CONTEXT_LENGTH not in WINDOW_CONFIGS:
        typer.echo(
            f"Warning: No predefined window configuration for context_length={CONTEXT_LENGTH}",
            err=True,
        )
        typer.echo("Using default 8-week configuration", err=True)
    config = WINDOW_CONFIGS.get(
        CONTEXT_LENGTH, WINDOW_CONFIGS[DEFAULT_WINDOW_CONFIG_KEY]
    )
    WINDOW_LENGTHS = config["lengths"]
    WINDOW_NAMES = config["names"]

    explanations_base = (
        PROJECT_ROOT / f"results_tabpfn/explanations_parallel/{CONTEXT_LENGTH}/tabpfn_explanations"
    )
    print(f"{explanations_base}.pkl")
    assert np.sum(WINDOW_LENGTHS) == CONTEXT_LENGTH

    target = "target"
    prediction_length = 24
    id_column = "id"
    timestamp_column = "timestamp"

    transnet_df = get_transnetbw_df()
    print(transnet_df)

    features = ["holiday", "temperature", "irradiance"]

    n_groups = len(features) + len(WINDOW_LENGTHS)
    group_ids = list(range(n_groups))
    group_names = WINDOW_NAMES + features

    explanations = []

    # Create feature transformer once (reuse across all predictions)
    base_features = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    feature_transformer = FeatureTransformer(base_features)

    # Start timing for explanation generation
    start_time = time.time()
    for day in tqdm(range(DAYS)):
        start_date = START_DATE + timedelta(days=day)
        predictions = {}
        for subset in powerset(group_ids):
            subset = tuple(subset)

            energy_context_df = transnet_df[transnet_df[timestamp_column] < start_date]
            energy_context_df = energy_context_df[-CONTEXT_LENGTH:].copy()
            energy_future_df = transnet_df[transnet_df[timestamp_column] >= start_date]
            energy_future_df = energy_future_df[:prediction_length].copy()
            ground_truth = np.array(energy_future_df[target])

            min_group = min(subset) if len(subset) > 0 else None

            if len(subset) == 0 or min_group >= len(WINDOW_LENGTHS):
                # Target feature is inactive - use mean prediction
                prediction = [np.mean(energy_context_df[target])] * prediction_length
            else:
                context_len = int(np.sum(WINDOW_LENGTHS[min_group:]))
                energy_context_df = energy_context_df[-context_len:]
                pos = 0
                for gi in range(min_group, len(WINDOW_LENGTHS)):
                    window_len = WINDOW_LENGTHS[gi]
                    if gi not in subset:
                        # set all target values that are not in the subset to None
                        vals = energy_context_df[target].values
                        vals[pos : pos + window_len] = None
                        energy_context_df[target] = vals
                    pos += window_len
                for gi in range(len(WINDOW_LENGTHS), len(group_ids)):
                    if gi not in subset:
                        feature = features[gi - len(WINDOW_LENGTHS)]
                        energy_context_df = energy_context_df.drop(columns=feature)
                        energy_future_df = energy_future_df.drop(columns=feature)
                if "holiday" in energy_context_df:
                    if (
                        np.mean(energy_context_df["holiday"]) == 0
                        and np.mean(energy_future_df["holiday"]) == 0
                    ) or (
                        np.mean(energy_context_df["holiday"]) == 1
                        and np.mean(energy_future_df["holiday"]) == 1
                    ):
                        energy_context_df = energy_context_df.drop(columns=["holiday"])
                        energy_future_df = energy_future_df.drop(columns=["holiday"])
                energy_context_df_cleaned = energy_context_df.copy()
                energy_context_df_cleaned.dropna(inplace=True)
                # Skip if we don't have enough data after masking
                if len(energy_context_df_cleaned) == 0:
                    prediction = [
                        np.mean(transnet_df[target].iloc[-CONTEXT_LENGTH:])
                    ] * prediction_length
                else:
                    # Convert to TimeSeriesDataFrame format
                    train_tsdf = TimeSeriesDataFrame(
                        energy_context_df_cleaned.rename(
                            columns={id_column: "item_id"}
                        ).set_index(["item_id", timestamp_column])
                    )

                    test_tsdf = TimeSeriesDataFrame(
                        energy_future_df.rename(
                            columns={id_column: "item_id"}
                        ).set_index(["item_id", timestamp_column])
                    )
                    test_tsdf_copy = test_tsdf.copy()
                    test_tsdf_copy["target"] = np.nan

                    # Transform features
                    train_tsdf_tf, test_tsdf_tf = feature_transformer.transform(
                        train_tsdf, test_tsdf_copy
                    )

                    # Remove unneeded features (hourly data only)
                    for df in [train_tsdf_tf, test_tsdf_tf]:
                        df.drop(
                            columns=[
                                c for c in UNNEEDED_HOURLY_FEATURES if c in df.columns
                            ],
                            inplace=True,
                        )

                    # Extract features and target
                    X_tr, y_tr = (
                        train_tsdf_tf.drop(columns=target),
                        train_tsdf_tf[target],
                    )
                    X_te = test_tsdf_tf.drop(columns=target)

                    # Fit and predict using ensemble
                    pipeline = TabPFNEnsemble(
                        device=DEVICE,
                        n_preprocessing_jobs=-1,
                    )
                    pipeline.fit(X_tr, y_tr)
                    prediction = pipeline.predict(X_te)
                    if not np.all(np.isfinite(prediction)):
                        # Calculate fallback mean from the original data
                        fallback_val = np.mean(
                            transnet_df[target].iloc[-CONTEXT_LENGTH:]
                        )
                        # Replace only the invalid values with the mean
                        prediction = np.where(
                            np.isfinite(prediction), prediction, fallback_val
                        )

            predictions[subset] = np.array(prediction)

        feature_impacts = {
            group_name: np.zeros(prediction_length) for group_name in group_names
        }
        for subset in predictions:
            prediction = predictions[subset]
            for i, gi in enumerate(subset):
                other_subset = tuple(subset[:i] + subset[i + 1 :])
                other_prediction = predictions[other_subset]
                marginal = prediction - other_prediction
                weight = (
                    factorial(n_groups - 1 - len(other_subset))
                    * factorial(len(other_subset))
                    / factorial(n_groups)
                )
                group_name = group_names[gi]
                feature_impacts[group_name] += weight * marginal

        base = predictions[tuple()]

        explanation = {
            "prediction": predictions[tuple(group_ids)],
            "ground_truth": ground_truth,
            "base": base,
            "predictions": predictions,
            "shap_values": feature_impacts,
            "context_df": energy_context_df,
            "future_df": energy_future_df,
            "date": str(start_date),
        }
        explanations.append(explanation)

    # End timing and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_day = total_time / DAYS

    explanations_file = f"{explanations_base}.pkl"
    with open(explanations_file, "wb") as f:
        pickle.dump(explanations, f)

    print(f"\nExplanations saved to '{explanations_file}'")
    print(f"Total days processed: {len(explanations)}")
    print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Average time per day: {avg_time_per_day:.2f} seconds")


if __name__ == "__main__":
    typer.run(main)
