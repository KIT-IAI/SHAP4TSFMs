# Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models

This repository contains the code accompanying our publication on an algorithm for efficient calculation of Shapley additive explanations (SHAP) for time-series foundation models and applying it to load forecasting.

Stay tuned for the upcoming publication:

> Matthias Hertel, Alexandra Nikoltchovska, Sebastian Pütz, Benjamin Schäfer, Ralf Mikut, Veit Hagenmeyer.
> *Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models.*
> Under Review (2026).

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- CUDA-capable GPU (recommended; CPU/MPS fallback available for TabPFN)

### Setup

1. Clone the repository:

```bash
git clone <https://github.com/KIT-IAI/SHAP4TSFMs>
cd shap4tsfms
```

2. Install all dependencies using uv:

```bash
uv sync
```

This installs all project dependencies defined in `pyproject.toml`, including PyTorch with the appropriate backend (CPU on macOS/Windows, CUDA 12.4 on Linux).

To also install the development dependencies (ipykernel, pdbpp, wandb):

```bash
uv sync --all-groups
```

## Usage

All scripts are run via `uv run` to use the project's virtual environment.

### Forecasting with Chronos-2

Generate load forecasts using the Chronos-2 foundation model:

```bash
uv run chronos-2-predict.py
```

Compute SHAP-based explanations for Chronos-2 forecasts:

```bash
uv run chronos-2-subseries.py
```

### Forecasting with TabPFN-TS

Run TabPFN-TS multivariate forecasting (requires `context_name` and `context_length` arguments):

```bash
uv run tabpfn-predict.py 1_year 8760
```

Run TabPFN univariate forecasting:

```bash
uv run python tabpfn-predict-univariate.py
```

Compute SHAP-based subseries explanations for TabPFN:

```bash
uv run tabpfn-ts-subseries.py
```

Optional arguments for `tabpfn-ts-subseries.py`:

| Argument            | Default      | Description                                                        |
|---------------------|--------------|--------------------------------------------------------------------|
| `--context-length`  | `8760`       | Context window length in hours (`1344`, `8064`, or `8760`)         |
| `--start-date`      | `2024-10-01` | Start date for the forecast period                                 |

### Evaluation and Visualization

Generate SHAP feature dependence plots and compute feature importance:

```bash
uv run evaluate_explanations.py --model chronos-2
```

| Argument  | Default     | Choices                    | Description              |
|-----------|-------------|----------------------------|--------------------------|
| `--model` | `chronos-2` | `tabpfn-ts`, `chronos-2`   | Model to evaluate        |

Generate monthly stacked SHAP waterfall plots:

```bash
uv run stacked_plot.py
```

Generate SHAP waterfall plots with exogenous weather overlays (monthly, 4-day panel, and single-day plots):

```bash
uv run stacked_plot_with_exogenous.py
```

## Data

The `data/` directory contains hourly time series for the TransnetBW control area (Baden-Württemberg, Germany), aggregated over four NUTS regions (DE11–DE14):

- `TransnetBW_Total_Load.csv` — electrical load
- `Air_Temperature_2m.csv` — air temperature (°C)
- `Global_Horizontal_Irradiance.csv` — solar irradiance (W/m²)
- `Wind_Speed_10m.csv` / `Wind_Speed_100m.csv` — wind speed (m/s)
- `Mean_Sea_Level_Pressure.csv` — pressure
- `Total_Precipitation.csv` — precipitation


## License

This project is licensed under the MIT License — see [LICENSE](LICENSE.MD) for details.
