import pickle
from datetime import datetime, timedelta
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_xkcd_colors():
    with open("xkcd_colors.txt") as f:
        lines = f.readlines()
    colors = [l.strip().split("\t")[1] for l in lines[1:] if len(l) > 0]
    return colors[::-1]


XKCD_COLORS = get_xkcd_colors()
PREDICTION_COLOR = XKCD_COLORS[5]
GROUND_TRUTH_COLOR = "black"
FEATURE_COLORS = {
    "base": "gray",
    "irradiance": XKCD_COLORS[11],
    "holiday": XKCD_COLORS[4],
    "temperature": XKCD_COLORS[8],
    "week-48--5": "#15df1c",
    "week-4--2": "#159f1a",
    "day-7--2": "#156017",
    "day-1": "#152015",
}


def get_color(feature):
    if feature in FEATURE_COLORS:
        return FEATURE_COLORS[feature]
    return None


def waterfall_plot(
    feature_impacts: Dict[str, np.ndarray],
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    start_date: datetime,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    pos_contribs = []
    neg_contribs = []

    for feature in feature_impacts:
        contrib = feature_impacts[feature]
        pos_contribs.append(np.clip(contrib, a_min=0, a_max=None))
        neg_contribs.append(np.clip(contrib, a_min=None, a_max=0))

    colors = [get_color(feature) for feature in feature_impacts]

    x = pd.date_range(start_date, periods=len(prediction), freq="h")
    ax.stackplot(x, pos_contribs, colors=colors, labels=list(feature_impacts))
    ax.stackplot(x, neg_contribs, colors=colors)
    ax.plot(x, prediction, color=PREDICTION_COLOR, label="prediction")
    ax.plot(x, ground_truth, color=GROUND_TRUTH_COLOR, label="target")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})
    ax.set_ylabel("SHAP value [MW]")


if __name__ == "__main__":
    # EXPLANATIONS_FILE = "chronos-2_subseries_explanations.pkl"
    EXPLANATIONS_FILE = "tabpfn-ts_subseries_explanations.pkl"
    START_DATE = datetime(2024, 10, 1)

    with open(EXPLANATIONS_FILE, "rb") as f:
        explanations = pickle.load(f)

    features = list(explanations[0]["shap_values"])

    shap_values = {ft: [] for ft in features}
    prediction = []
    ground_truth = []

    for explanation in explanations:
        for ft in features:
            shap_values[ft].extend(explanation["shap_values"][ft])
        prediction.extend(explanation["prediction"])
        ground_truth.extend(explanation["ground_truth"])

    for ft in features:
        shap_values[ft] = np.array(shap_values[ft])
    prediction = np.array(prediction)

    fig, axs = plt.subplots(12, 1, figsize=(16, 24))
    month_lengths = [31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30]
    for i, length in enumerate(month_lengths):
        start_index = int(np.sum(month_lengths[:i]) * 24)
        end_index = start_index + length * 24
        month_shaps = {ft: shap_values[ft][start_index:end_index] for ft in features}
        month_pred = prediction[start_index:end_index]
        month_gt = ground_truth[start_index:end_index]
        start_date = START_DATE + timedelta(hours=start_index)
        waterfall_plot(month_shaps, month_pred, month_gt, start_date, ax=axs[i])
    plt.savefig("plots/monthly_shaps.pdf", bbox_inches="tight")
    # plt.show()

    ft_importance = {ft: np.sum(np.abs(shap_values[ft])) for ft in features}
    total_importance = sum(ft_importance[ft] for ft in features)
    print("Feature importances:")
    for ft in features:
        print(ft, ft_importance[ft], ft_importance[ft] / total_importance)
