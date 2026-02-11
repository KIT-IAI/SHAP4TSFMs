import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline
from tqdm import tqdm
from workalendar.europe import BadenWurttemberg


def load_pipeline(file):
    if os.path.exists(file):
        with open(file, "rb") as f:
            pipeline: Chronos2Pipeline = pickle.load(f)
    else:
        pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            "s3://autogluon/chronos-2/", device_map="cuda"
        )
        with open(file, "wb") as f:
            pickle.dump(pipeline, f)
    return pipeline


TRANSNET_FEATURES = [
    ("temperature", "data/Air_Temperature_2m.csv"),
    ("irradiance", "data/Global_Horizontal_Irradiance.csv"),
]


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


if __name__ == "__main__":
    CONTEXT_LENGTH = int(sys.argv[1])
    PIPELINE_FILE = "pipeline.pkl"

    N_VARIABLES = 4
    START_DATE = datetime(2024, 10, 1)
    END_DATE = datetime(2025, 9, 30)
    STRIDE = 1

    pipeline = load_pipeline(PIPELINE_FILE)

    target = "target"  # Column name containing the values to forecast (energy prices)
    prediction_length = 24  # Number of hours to forecast ahead
    id_column = "id"  # Column identifying different time series (countries/regions)
    timestamp_column = "timestamp"  # Column containing datetime information

    transnet_df = get_transnetbw_df()
    print(transnet_df)

    predictions = []
    ground_truths = []

    features = ["target", "holiday", "temperature", "irradiance"]

    total_hours = ((END_DATE - START_DATE).days + 1) * 24 - prediction_length + 1
    n_predictions = total_hours // STRIDE
    print(n_predictions)

    for pred_i in tqdm(range(n_predictions)):
        start_date = START_DATE + timedelta(hours=pred_i * STRIDE)

        energy_context_df = transnet_df[transnet_df[timestamp_column] < start_date]
        energy_context_df = energy_context_df[-CONTEXT_LENGTH:]
        energy_future_df = transnet_df[transnet_df[timestamp_column] >= start_date]
        energy_future_df = energy_future_df[:prediction_length]
        ground_truth = np.array(energy_future_df[target])
        energy_future_df = energy_future_df.drop(columns=target)

        prediction = pipeline.predict_df(
            energy_context_df,
            future_df=energy_future_df,
            prediction_length=prediction_length,
            quantile_levels=[0.5],  # [0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
        )["predictions"]

        predictions.append(prediction)
        ground_truths.append(ground_truth)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    print(predictions.shape)
    print(ground_truths.shape)

    mae = np.mean(np.abs(predictions - ground_truths))
    mse = np.mean(np.square(predictions - ground_truths))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - ground_truths) / ground_truths))

    print("mae:", mae)
    print("mse:", mse)
    print("rmse:", rmse)
    print("mape:", mape)
