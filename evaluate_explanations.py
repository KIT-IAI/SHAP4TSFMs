import argparse
import pickle
from datetime import datetime, timedelta
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def get_feature_importance(explanations: List[Dict]):
    features = explanations[0]["shap_values"].keys()
    importance = {feature: 0 for feature in features}
    for feature in features:
        for explanation in explanations:
            importance[feature] += np.sum(np.abs(explanation["shap_values"][feature]))
    total_importance = sum(importance.values())
    for feature in importance:
        importance[feature] /= total_importance
    return importance


def scatter(
    x,
    y,
    c,
    xlabel,
    clabel,
    cmap,
    window,
    model,
    holiday=None,
    diagonal=None,
    ylabel=None,
    ylimits=None,
    save_label=None,
):
    """
    Generates a scatter plot with a line for the sliding mean.
    For each x value, a sliding mean is computed on all points within (x - window, x + window).

    :param x: x values
    :param y: y values
    :param c: color values
    :param xlabel: x-axis label
    :param clabel: label of the colormap
    :param cmap: the colormap for the dot colors
    :param window: window size for the sliding mean
    :param holiday: optional vector of same length as x and y, with 1 if it is a holiday and 0 else (holidays get a different color)
    :return:
    """
    x = np.array(x)
    y = np.array(y)
    c = np.array(c)
    if holiday is None:
        plt.scatter(x, y, c=c, s=1, cmap=cmap)
    else:
        plt.scatter(
            x[holiday == 1],
            y[holiday == 1],
            c=c[holiday == 1],
            s=1,
            cmap="Reds",
            vmin=min(c),
            vmax=max(c),
        )
        plt.colorbar(label="past load [MW] (holiday)")
        plt.scatter(
            x[holiday == 0],
            y[holiday == 0],
            c=c[holiday == 0],
            s=1,
            cmap=cmap,
            vmin=min(c),
            vmax=max(c),
        )
    if window is not None:
        srt_idx = np.argsort(x)
        x = x[srt_idx]
        y = y[srt_idx]
        smoothed_y = []
        left_ptr = 0
        right_ptr = 0
        for val in x:
            while x[left_ptr] < val - window:
                left_ptr += 1
            while right_ptr < len(x) and x[right_ptr] < val + window:
                right_ptr += 1
            smoothed_y.append(np.mean(y[left_ptr:right_ptr]))
        plt.plot(x, smoothed_y, color="red")
    if diagonal is not None:
        plt.plot(diagonal, diagonal, color="black")
    if xlabel == "Hour of week":
        plt.xticks(np.arange(0, 168, 24))
    else:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("SHAP value [MW]")

    plt.ylim(ylimits)
    plt.colorbar(label=clabel)
    plt.tight_layout()
    if save_label is not None:
        plt.savefig(f"plots/{model}_{save_label}.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"plots/{model}_{xlabel}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SHAP feature dependence plots and compute feature importance."
    )
    parser.add_argument(
        "--model",
        choices=["tabpfn-ts", "chronos-2"],
        default="chronos-2",
        help="Model to evaluate (default: chronos-2)",
    )
    args = parser.parse_args()

    MODELS = ["tabpfn-ts", "chronos-2"]
    START_DATE = datetime(2024, 10, 1)

    # Dictionary to hold all loaded data
    all_expls = {}

    for model_name in MODELS:
        filename = f"{model_name}_subseries_explanations.pkl"
        with open(filename, "rb") as f:
            all_expls[model_name] = pickle.load(f)

    feature_limits = {
        feat: (
            min(
                exp["shap_values"][feat].min()
                for data in all_expls.values()
                for exp in data
            )
            * 1.1,
            max(
                exp["shap_values"][feat].max()
                for data in all_expls.values()
                for exp in data
            )
            * 1.1,
        )
        for feat in ["holiday", "temperature", "irradiance"]
    }

    model = args.model
    START_DATE = datetime(2024, 10, 1)

    explanations = all_expls[model]
    features = list(explanations[0]["shap_values"])

    # feature importance
    importance = get_feature_importance(explanations)
    for feature in features:
        print(feature, importance[feature])

    # plot predictions and local explanations
    ground_truth = np.array([exp["ground_truth"] for exp in explanations]).flatten()
    prediction = np.array([exp["prediction"] for exp in explanations]).flatten()
    base = np.array([exp["base"] for exp in explanations]).flatten()

    index = [START_DATE + timedelta(hours=i) for i in range(len(ground_truth))]

    # plt.plot(index, ground_truth, label="target")
    # plt.plot(index, prediction, label="prediction")
    # plt.plot(index, base, label="base")
    plt.plot(index, ground_truth, label="target")
    plt.plot(index, prediction, label="prediction")
    plt.plot(index, base, label="base")
    for feature in features:
        shap_values = np.array(
            [exp["shap_values"][feature] for exp in explanations]
        ).flatten()
        plt.plot(index, shap_values, label=f"{feature} SHAP values")
    plt.xlabel("Date")
    plt.ylabel("Load [MW]")
    plt.legend()
    plt.show()

    # holiday feature dependence plot
    feature = "holiday"
    ft_values = []
    shap_values = []
    hod_values = []
    dow_values = []
    for explanation in explanations:
        previous_holiday = (
            explanation["context_df"]["holiday"].iloc[-1] == 1
            and explanation["context_df"]["timestamp"].iloc[-1].weekday() != 6
        )
        is_holiday = (
            explanation["future_df"]["holiday"].iloc[0] == 1
            and explanation["future_df"]["timestamp"].iloc[0].weekday() != 6
        )
        is_monday = explanation["future_df"]["timestamp"].iloc[0].weekday() == 0
        is_sunday = explanation["future_df"]["timestamp"].iloc[0].weekday() == 6
        if not previous_holiday and not is_holiday and is_sunday:
            ft_values.extend([0] * 24)
        elif not previous_holiday and not is_holiday and is_monday:
            ft_values.extend([1] * 24)
        elif (
            not previous_holiday and not is_holiday and not is_monday and not is_sunday
        ):
            ft_values.extend([2] * 24)
        elif previous_holiday and not is_holiday:
            ft_values.extend([3] * 24)
        elif not previous_holiday and is_holiday:
            ft_values.extend([4] * 24)
        else:
            ft_values.extend([5] * 24)
        shap_values.extend(explanation["shap_values"][feature])
        hod_values.extend(np.arange(24))
        dow = datetime.strptime(explanation["date"], "%Y-%m-%d %H:%M:%S").weekday()
        dow_values.extend([dow] * 24)
    plt.scatter(
        ft_values + np.random.uniform(-0.3, 0.3, size=len(ft_values)),
        shap_values,
        c=hod_values,
        cmap="twilight",
        s=1,
    )
    plt.xticks(
        [0, 1, 2, 3, 4, 5],
        [
            "Sa/Su ",
            " Su/Mo",
            r"$\circ$/$\circ$",
            r"$\bullet$/$\circ$",
            r"$\circ$/$\bullet$",
            r"$\bullet$/$\bullet$",
        ],
    )
    plt.xlabel(r"Day type (previous/current): $\circ$=workday, $\bullet$=holiday")
    plt.ylabel("SHAP value [MW]")
    plt.ylim(feature_limits["holiday"])
    plt.colorbar(label="Hour of day")
    plt.tight_layout()
    plt.savefig(f"plots/{model}_{feature}.pdf")
    plt.show()

    # feature dependence plots
    for feature in ["temperature", "irradiance"]:
        ft_values = []
        ft_diff_values = []
        shap_values = []
        hod_values = []
        dow_values = []
        how_values = []
        holiday_values = []
        for explanation in explanations:
            if feature == "target":
                ft_values.extend(explanation["context_df"]["target"].iloc[-24:])
            else:
                ft_values.extend(explanation["future_df"][feature])
                ft_diff_values.extend(
                    np.array(explanation["future_df"][feature])
                    - np.array(explanation["context_df"][feature][-24:])
                )
            shap_values.extend(explanation["shap_values"][feature])
            hod_values.extend(np.arange(24))
            dow = datetime.strptime(explanation["date"], "%Y-%m-%d %H:%M:%S").weekday()
            dow_values.extend([dow] * 24)
            how_values.extend(np.arange(dow * 24, dow * 24 + 24) % 168)
            holiday_values.extend(
                explanation["future_df"]["holiday"]
                * (
                    1
                    if explanation["future_df"]["timestamp"].iloc[0].weekday() != 6
                    else 0
                )
            )

        feature_units = {"temperature": r"[$^\circ$C]", "irradiance": r"[W/m$^2$]"}
        scatter(
            x=how_values if feature == "target" else ft_values,
            y=shap_values,
            c=ft_values if feature == "target" else hod_values,
            xlabel="Hour of week"
            if feature == "target"
            else f"{feature.capitalize()} {feature_units[feature]}",
            cmap="Blues" if feature == "target" else "twilight",
            clabel="Past load [MW]" if feature == "target" else "Hour of day",
            window=None,
            model=model,
            holiday=np.array(holiday_values) if feature == "target" else None,
            save_label=feature,
        )

        if feature != "target":
            scatter(
                x=ft_diff_values,
                y=shap_values,
                c=hod_values,
                xlabel=f"{feature.capitalize()} difference {feature_units[feature]}",
                cmap="twilight",
                clabel="Hour of day",
                window=None,
                model=model,
                ylimits=feature_limits[feature],
                save_label=f"{feature}_difference",
            )

        if feature == "temperature":
            scatter(
                x=ft_values,
                y=shap_values,
                c=ft_diff_values,
                cmap="coolwarm",
                xlabel=f"{feature.capitalize()} {feature_units[feature]}",
                clabel=f"{feature.capitalize()} difference [°C]",
                window=None,
                model=model,
                ylimits=feature_limits[feature],
                save_label=f"{feature}",
            )

            scatter(
                x=ft_diff_values,
                y=shap_values,
                c=ft_values,
                cmap="coolwarm",
                xlabel=f"{feature.capitalize()} difference [°C]",
                clabel="Temperature [°C]",
                window=None,
                model=model,
                ylimits=feature_limits[feature],
                save_label=f"{feature}_difference",
            )

            values_day_before = (np.asarray(ft_values) - np.asarray(ft_diff_values),)

            scatter(
                x=values_day_before,
                y=ft_values,
                c=shap_values,
                cmap="coolwarm",
                xlabel="Temperature day before [°C]",
                ylabel="Temperature [°C]",
                clabel="SHAP value [MW]",
                window=None,
                model=model,
                diagonal=(-8, 35),
                save_label=f"{feature}_day_before",
            )
