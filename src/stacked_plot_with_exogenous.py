import pickle
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
from workalendar.europe import BadenWurttemberg

GROUND_TRUTH_COLOR = "black"
PREDICTION_COLOR = "#56B4E9"
FEATURE_COLORS = {
    "base": "gray",
    "irradiance": "#E69F00",
    "holiday": "#CC79A7",
    "temperature": "#D55E00",
    "week-48--5": "#4DCCB3",
    "week-52--5": "#4DCCB3",
    "week-4--2": "#00B893",
    "day-7--2": "#009E73",
    "day-1": "#006B4D",
}

FEATURE_DISPLAY_NAMES = {
    "base": "Base",
    "irradiance": "Irradiance",
    "holiday": "Holiday",
    "temperature": "Temperature",
    "week-48--5": "Long-term",
    "week-52--5": "Long-term",
    "week-4--2": "Intermediate",
    "day-7--2": "Short-term",
    "day-1": "Last day",
}

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


def _add_exogenous_axis(
    ax,
    x,
    data,
    feature_name,
    ylabel,
    global_min,
    global_max,
    left_ymin,
    left_ymax,
    linestyle,
    spine_offset=0,
    compression_factor=0.5,
    tick_labelsize=12,
    ylabel_fontsize=16,
    ylabel_labelpad=-42,
    show_yaxis=True,
):
    ax_exo = ax.twinx()
    if spine_offset > 0:
        ax_exo.spines["right"].set_position(("outward", spine_offset))

    ax_exo.plot(
        x,
        data,
        color=FEATURE_COLORS[feature_name],
        label=feature_name,
        linewidth=2,
        alpha=0.8,
        linestyle=linestyle,
    )
    if show_yaxis:
        ax_exo.set_ylabel(
            ylabel,
            color=FEATURE_COLORS[feature_name],
            fontsize=ylabel_fontsize,
            labelpad=ylabel_labelpad,
        )
        ax_exo.tick_params(
            axis="y", labelcolor=FEATURE_COLORS[feature_name], labelsize=tick_labelsize
        )
        ax_exo.spines["right"].set_color(FEATURE_COLORS[feature_name])
        ax_exo.spines["right"].set_linewidth(2)
    else:
        ax_exo.set_ylabel("")
        ax_exo.tick_params(
            axis="y", labelleft=False, labelright=False, left=False, right=False
        )
        ax_exo.spines["right"].set_visible(False)

    from matplotlib.ticker import FuncFormatter

    ax_exo.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:>5.0f}"))

    if left_ymin < 0 < left_ymax:
        ratio = abs(left_ymin) / (left_ymax - left_ymin)
    else:
        ratio = 0 if left_ymin >= 0 else 1

    if global_min is not None and global_max is not None:
        data_range = global_max - global_min
        expansion = data_range * (1 / compression_factor - 1) / 2
        expanded_min, expanded_max = global_min - expansion, global_max + expansion

        if 0 < ratio < 1:
            range_below_zero = max(
                abs(expanded_min), ratio * (expanded_max - expanded_min) / (1 - ratio)
            )
            ax_exo.set_ylim([-range_below_zero, range_below_zero * (1 - ratio) / ratio])
        else:
            ax_exo.set_ylim([expanded_min, expanded_max])
    return ax_exo


def waterfall_plot(
    feature_impacts: dict[str, np.ndarray],
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    start_date: datetime,
    irradiance_data: np.ndarray = None,
    temperature_data: np.ndarray = None,
    irr_min: float = None,
    irr_max: float = None,
    temp_min: float = None,
    temp_max: float = None,
    holiday_data: np.ndarray = None,
    ax=None,
    show_legend=True,
    tick_labelsize=12,
    ylabel_fontsize=16,
    show_exogenous_yaxis=True,
    shared_ylim=None,
    show_ground_truth=True,
):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    pos_contribs = [np.clip(v, 0, None) for v in feature_impacts.values()]
    neg_contribs = [np.clip(v, None, 0) for v in feature_impacts.values()]
    colors = [FEATURE_COLORS.get(f) for f in feature_impacts]
    display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_impacts]

    x = pd.date_range(start_date, periods=len(prediction), freq="h")
    ax.stackplot(x, pos_contribs, colors=colors, labels=display_labels)
    ax.stackplot(x, neg_contribs, colors=colors)
    ax.plot(
        x, prediction, color=PREDICTION_COLOR, label="Predicted Load", linewidth=1.5
    )
    if show_ground_truth:
        ax.plot(
            x,
            ground_truth,
            color=GROUND_TRUTH_COLOR,
            label="Target Load",
            linewidth=1.5,
        )
    ax.set_ylabel("SHAP value [MW]", fontsize=ylabel_fontsize)
    ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.tick_params(axis="x", labelsize=tick_labelsize)
    ax.set_xlim(x[0], x[-1])

    # Align y-axis labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:>5.0f}"))

    # Color x-axis tick labels and background for holidays
    if holiday_data is not None:
        for tick_label in ax.get_xticklabels():
            tick_date = pd.to_datetime(tick_label.get_text())
            if pd.notna(tick_date):
                idx = (tick_date - x[0]).days * 24 + (tick_date - x[0]).seconds // 3600
                if 0 <= idx < len(holiday_data) and holiday_data[idx] > 0:
                    tick_label.set_color(FEATURE_COLORS["holiday"])

        holiday_color = to_rgba(FEATURE_COLORS["holiday"])
        light_holiday_color = (*holiday_color[:3], 0.15)

        i = 0
        while i < len(holiday_data):
            if holiday_data[i] > 0:
                start_idx = i
                while i < len(holiday_data) and holiday_data[i] > 0:
                    i += 1
                ax.axvspan(x[start_idx], x[i - 1], color=light_holiday_color, zorder=0)
            else:
                i += 1

    if shared_ylim is not None:
        ax.set_ylim(shared_ylim)
        left_ymin, left_ymax = shared_ylim
    else:
        left_ymin, left_ymax = ax.get_ylim()

    # Add exogenous axes with appropriate spine offsets
    # irr_spine_offset = 35 if irradiance_data is not None else None
    # temp_spine_offset = (
    #     (105 if irradiance_data is not None else 35)
    #     if temperature_data is not None
    #     else None
    # )
    irr_spine_offset = 50 if irradiance_data is not None else None
    temp_spine_offset = (
        (125 if irradiance_data is not None else 50)
        if temperature_data is not None
        else None
    )
    # modify this for 4-day-panel
    ylabel_labelpad = -70 if ylabel_fontsize >= 14 else -30

    if irradiance_data is not None:
        _add_exogenous_axis(
            ax,
            x,
            irradiance_data,
            "irradiance",
            "Irradiance [W/m²]",
            irr_min,
            irr_max,
            left_ymin,
            left_ymax,
            linestyle="--",
            spine_offset=irr_spine_offset,
            # compression_factor=0.45, for whole year plot
            compression_factor=0.55,
            tick_labelsize=tick_labelsize,
            ylabel_fontsize=ylabel_fontsize,
            ylabel_labelpad=ylabel_labelpad,
            show_yaxis=show_exogenous_yaxis,
        )

    if temperature_data is not None:
        _add_exogenous_axis(
            ax,
            x,
            temperature_data,
            "temperature",
            "Temperature [°C]",
            temp_min,
            temp_max,
            left_ymin,
            left_ymax,
            linestyle=":",
            spine_offset=temp_spine_offset,
            # compression_factor=0.68, for whole year plot
            compression_factor=0.88,
            tick_labelsize=tick_labelsize,
            ylabel_fontsize=ylabel_fontsize,
            # add +7 for 4-day-panel
            ylabel_labelpad=ylabel_labelpad + 7,
            show_yaxis=show_exogenous_yaxis,
        )

    if show_legend:
        fig_width = ax.figure.get_size_inches()[0]
        rightmost_offset = temp_spine_offset or irr_spine_offset or 0
        legend_offset_pts = rightmost_offset + (60 if fig_width < 15 else 40)
        legend_x_position = 1.0 + legend_offset_pts / 72 / fig_width
        ax.legend(
            loc="center left",
            bbox_to_anchor=(legend_x_position, 0.5),
            prop={"size": 12},
        )

    return ax


def plot_n_days(
    shap_values,
    prediction,
    ground_truth,
    start_date,
    irradiance_full=None,
    temperature_full=None,
    holiday_full=None,
    n_days=3,
    plot_start_date=None,
    ax=None,
    save_path=None,
    show_legend=True,
    tick_labelsize=12,
    ylabel_fontsize=16,
    exogenous_filter=None,
    show_exogenous_yaxis=True,
    shared_ylim=None,
    context_windows=None,
    show_ground_truth=None,
):
    start_idx = (
        int((plot_start_date - start_date).total_seconds() / 3600)
        if plot_start_date
        else 0
    )
    end_idx = start_idx + n_days * 24

    # Filter context windows if specified
    if context_windows is not None:
        filtered_shap = {
            ft: shap_values[ft] for ft in shap_values if ft in context_windows
        }
    else:
        filtered_shap = shap_values

    shap_values_slice = {
        ft: filtered_shap[ft][start_idx:end_idx] for ft in filtered_shap
    }
    prediction_slice = prediction[start_idx:end_idx]
    ground_truth_slice = np.array(ground_truth)[start_idx:end_idx]
    slice_start_date = start_date + timedelta(hours=start_idx)

    def extract_exo(data, feature):
        if data is None or exogenous_filter not in [None, feature]:
            return None, None, None
        sliced = data[start_idx:end_idx]
        return sliced, sliced.min(), sliced.max()

    irradiance_data, irr_min, irr_max = extract_exo(irradiance_full, "irradiance")
    temperature_data, temp_min, temp_max = extract_exo(temperature_full, "temperature")
    holiday_data = (
        holiday_full[start_idx:end_idx]
        if holiday_full is not None and exogenous_filter in [None, "holiday"]
        else None
    )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(11, 5))

    waterfall_plot(
        shap_values_slice,
        prediction_slice,
        ground_truth_slice,
        slice_start_date,
        irradiance_data=irradiance_data,
        temperature_data=temperature_data,
        irr_min=irr_min,
        irr_max=irr_max,
        temp_min=temp_min,
        temp_max=temp_max,
        holiday_data=holiday_data,
        ax=ax,
        show_legend=show_legend,
        tick_labelsize=tick_labelsize,
        ylabel_fontsize=ylabel_fontsize,
        show_exogenous_yaxis=show_exogenous_yaxis,
        shared_ylim=shared_ylim,
        show_ground_truth=show_ground_truth,
    )

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")

    return ax


def create_figure_with_legend(subplots_config, save_path):
    """Helper to create figures with shared legends."""
    fig, axs = plt.subplots(
        *subplots_config["shape"], figsize=subplots_config["figsize"]
    )
    axs_flat = np.atleast_1d(axs).flatten()

    for i, plot_fn in enumerate(subplots_config["plots"]):
        plot_fn(axs_flat[i])

    # Align y-axis labels across all subplots
    fig.align_ylabels(axs_flat)

    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        fontsize=subplots_config.get("legend_fontsize", 12),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, bbox_inches="tight")
    return fig, axs


if __name__ == "__main__":
    models = ["chronos-2", "tabpfn-ts"]
    for model in models:
        print(model)
        EXPLANATIONS_FILE = PROJECT_ROOT / f"{model}_subseries_explanations.pkl"
        START_DATE = datetime(2024, 10, 1)

        with open(EXPLANATIONS_FILE, "rb") as f:
            explanations = pickle.load(f)

        features = list(explanations[0]["shap_values"])
        print(features)
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

        transnet_df = get_transnetbw_df()
        start_idx = transnet_df[transnet_df["timestamp"] == START_DATE].index[0]
        end_idx = start_idx + len(prediction) * 4
        irradiance_full = transnet_df["irradiance"].loc[start_idx:end_idx].values
        temperature_full = transnet_df["temperature"].loc[start_idx:end_idx].values
        holiday_full = transnet_df["holiday"].loc[start_idx:end_idx].values

        # Monthly plots
        month_lengths = [31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30]
        monthly_plots = [
            lambda ax, i=i, length=length: plot_n_days(
                shap_values,
                prediction,
                ground_truth,
                START_DATE,
                irradiance_full,
                temperature_full,
                holiday_full,
                n_days=length,
                plot_start_date=START_DATE + timedelta(days=sum(month_lengths[:i])),
                ax=ax,
                show_legend=False,
                tick_labelsize=12,
                ylabel_fontsize=14,
                show_ground_truth=True,
            )
            for i, length in enumerate(month_lengths)
        ]

        create_figure_with_legend(
            {
                "shape": (12, 1),
                "figsize": (18, 22),
                "plots": monthly_plots,
            },
            PROJECT_ROOT / f"plots/{model}_monthly_shaps.pdf",
        )

        # 4-day panel plots
        dates = {
            "last-day": datetime(2025, 1, 6),
            "temperature": datetime(2025, 2, 19),
            "holiday": datetime(2025, 4, 30),
            "irradiance": datetime(2025, 8, 19),
        }

        # Pre-compute global y-axis limits across all panels
        n_days = 3
        global_ymin, global_ymax = 0, 0
        gt_arr = np.array(ground_truth)
        for date in dates.values():
            start_idx = int((date - START_DATE).total_seconds() / 3600)
            end_idx = start_idx + n_days * 24
            global_ymax = max(
                global_ymax,
                prediction[start_idx:end_idx].max(),
                gt_arr[start_idx:end_idx].max(),
            )
            neg_sum = sum(
                np.clip(shap_values[ft][start_idx:end_idx], None, 0)
                for ft in shap_values
            )
            global_ymin = min(global_ymin, neg_sum.min())
        shared_ylim = (global_ymin * 1.05, global_ymax * 1.05)

        panel_plots = []
        for idx, (date_key, date) in enumerate(dates.items()):
            exo_filter = "none" if date_key == "last-day" else date_key

            def make_plot(
                ax, date=date, exo_filter=exo_filter, idx=idx, ylim=shared_ylim
            ):
                result_ax = plot_n_days(
                    shap_values,
                    prediction,
                    ground_truth,
                    START_DATE,
                    irradiance_full,
                    temperature_full,
                    holiday_full,
                    n_days=3,
                    plot_start_date=date,
                    ax=ax,
                    show_legend=False,
                    exogenous_filter=exo_filter,
                    shared_ylim=ylim,
                    tick_labelsize=14,
                    ylabel_fontsize=18,
                    show_ground_truth=True,
                )
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 31)))

                ax.set_xlabel(
                    ["(a)", "(b)", "(c)", "(d)"][idx], fontsize=16, fontweight="bold"
                )
                return result_ax

            panel_plots.append(make_plot)

        create_figure_with_legend(
            {
                "shape": (2, 2),
                "figsize": (20, 12),
                "plots": panel_plots,
                "legend_fontsize": 14,
            },
            PROJECT_ROOT / f"plots/{model}_4days_panel.pdf",
        )

        # Plot for Figure 1
        one_day_date = datetime(2025, 5, 1)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_n_days(
            shap_values,
            prediction,
            ground_truth,
            START_DATE,
            irradiance_full,
            temperature_full,
            holiday_full,
            n_days=3,
            plot_start_date=one_day_date,
            ax=ax,
            show_legend=False,
            tick_labelsize=14,
            ylabel_fontsize=16,
            exogenous_filter="none",
            context_windows=[
                "week-48--5",
                "week-52--5",
                "day-7--2",
                "day-1",
                "holiday",
                "temperature",
            ],  # Filter to show only these
            show_ground_truth=None,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 2, 3, 4]))
        # Add schematic legend with renamed covariates and custom order
        handles, labels = ax.get_legend_handles_labels()
        print(f"Original labels: {labels}")
        label_map = {
            "Short-term": "Intermediate",
            "Last day": "Short-term",
            "Temperature": "Covariate 1",
            "Holiday": "Covariate 2",
        }
        # Build dict of label -> handle (first occurrence wins for duplicates)
        label_to_handle = {}
        for handle, label in zip(handles, labels):
            mapped_label = label_map.get(label, label)
            if mapped_label not in label_to_handle:
                label_to_handle[mapped_label] = handle
        print(f"Mapped labels: {list(label_to_handle.keys())}")
        # Define exact order for legend (row1: first 3, row2: last 3)
        order = [
            "Long-term",
            "Covariate 1",
            "Intermediate",
            "Covariate 2",
            "Short-term",
            "Predicted Load",
        ]
        sorted_handles = [
            label_to_handle[lbl] for lbl in order if lbl in label_to_handle
        ]
        sorted_labels = [lbl for lbl in order if lbl in label_to_handle]
        print(f"Final sorted labels: {sorted_labels}")
        ax.legend(
            sorted_handles,
            sorted_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / f"plots/{model}_one_day.pdf", bbox_inches="tight")
        plt.close()
        # One day plot end

        # Feature importance
        ft_importance = {ft: np.sum(np.abs(shap_values[ft])) for ft in features}
        total_importance = sum(ft_importance.values())
        print("Feature importances:")
        for ft in features:
            print(ft, ft_importance[ft], ft_importance[ft] / total_importance)
