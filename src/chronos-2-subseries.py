import os
import pickle
from datetime import datetime, timedelta
from math import factorial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


TRANSNET_FEATURES = [
    ("temperature", PROJECT_ROOT / "data/Air_Temperature_2m.csv"),
    ("irradiance", PROJECT_ROOT / "data/Global_Horizontal_Irradiance.csv"),
]


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


if __name__ == "__main__":
    PIPELINE_FILE = "pipeline.pkl"
    EXPLANATIONS_FILE = PROJECT_ROOT / "chronos-2_subseries_explanations.pkl"

    N_VARIABLES = 4
    CONTEXT_LENGTH = 168 * 48
    START_DATE = datetime(2024, 10, 1)
    DAYS = 365
    WINDOW_LENGTHS = [168 * 44, 168 * 3, 24 * 6, 24]
    WINDOW_NAMES = ["week-48--5", "week-4--2", "day-7--2", "day-1"]

    assert np.sum(WINDOW_LENGTHS) == CONTEXT_LENGTH

    pipeline = load_pipeline(PIPELINE_FILE)

    target = "target"
    prediction_length = 24
    id_column = "id"
    timestamp_column = "timestamp"

    transnet_df = get_transnetbw_df()
    print(transnet_df)

    predictions = []
    ground_truths = []

    features = ["holiday", "temperature", "irradiance"]

    n_groups = len(features) + len(WINDOW_LENGTHS)
    group_ids = list(range(n_groups))
    group_names = WINDOW_NAMES + features

    explanations = []

    for day in tqdm(range(DAYS)):
        start_date = START_DATE + timedelta(days=day)

        predictions = {}

        for subset in powerset(group_ids):
            subset = tuple(subset)

            energy_context_df = transnet_df[transnet_df[timestamp_column] < start_date]
            energy_context_df = energy_context_df[-CONTEXT_LENGTH:]
            energy_future_df = transnet_df[transnet_df[timestamp_column] >= start_date]
            energy_future_df = energy_future_df[:prediction_length]
            ground_truth = np.array(energy_future_df[target])
            energy_future_df = energy_future_df.drop(columns=target)

            min_group = min(subset) if len(subset) > 0 else None

            if len(subset) == 0 or min_group >= len(
                WINDOW_LENGTHS
            ):  # load feature is inactive
                prediction = [np.mean(energy_context_df[target])] * prediction_length
            else:
                context_len = int(np.sum(WINDOW_LENGTHS[min_group:]))
                energy_context_df = energy_context_df[-context_len:]
                pos = 0
                for gi in range(min_group, len(WINDOW_LENGTHS)):
                    window_len = WINDOW_LENGTHS[gi]
                    if gi not in subset:
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

                prediction = pipeline.predict_df(
                    energy_context_df,
                    future_df=energy_future_df,
                    prediction_length=prediction_length,
                    quantile_levels=[0.5],
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    target=target,
                )["predictions"]

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

    with open(EXPLANATIONS_FILE, "wb") as f:
        pickle.dump(explanations, f)
