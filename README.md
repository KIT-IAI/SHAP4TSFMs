# Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models

This repository contains the code accompanying our publication on an algorithm for efficient calculation of Shapley additive explanations (SHAP) for time-series foundation models and applying it to load forecasting.

Stay tuned for the upcoming publication:

> Matthias Hertel, Alexandra Nikoltchovska, Sebastian Pütz, Benjamin Schäfer, Ralf Mikut, Veit Hagenmeyer.
> *Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models.*
> In: Proceedings of the 17th ACM International Conference on Future and Sustainable Energy Systems (e-Energy '26). ACM, 2026.
> DOI: [10.1145/3744255.3811724](https://doi.org/10.1145/3744255.3811724) — Preprint: [arXiv:2512.20514](https://doi.org/10.48550/arXiv.2512.20514)

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- CUDA-capable GPU (recommended; CPU/MPS fallback available for TabPFN)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/KIT-IAI/SHAP4TSFMs
cd SHAP4TSFMs
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

All scripts are located in `src/` and run via `uv run` to use the project's virtual environment.

### Forecasting with Chronos-2

Generate load forecasts using the Chronos-2 foundation model (requires `context_length` argument):

```bash
uv run src/chronos-2-predict.py 8.064
```

Compute SHAP-based explanations for Chronos-2 forecasts:

```bash
uv run src/chronos-2-subseries.py
```

### Forecasting with TabPFN-TS

Run TabPFN-TS multivariate forecasting (requires `context_name` and `context_length` arguments):

```bash
uv run src/tabpfn-predict.py 1_year 8760
```

Run TabPFN univariate forecasting:

```bash
uv run src/tabpfn-predict-univariate.py
```

Compute SHAP-based subseries explanations for TabPFN:

```bash
uv run src/tabpfn-ts-subseries.py
```

Optional arguments for `tabpfn-ts-subseries.py`:

| Argument            | Default      | Description                                                        |
|---------------------|--------------|--------------------------------------------------------------------|
| `--context-length`  | `8760`       | Context window length in hours (`1344`, `8064`, or `8760`)         |
| `--start-date`      | `2024-10-01` | Start date for the forecast period                                 |

### Evaluation and Visualization

Generate SHAP feature dependence plots and compute feature importance:

```bash
uv run src/evaluate_explanations.py --model chronos-2
```

| Argument  | Default     | Choices                    | Description              |
|-----------|-------------|----------------------------|--------------------------|
| `--model` | `chronos-2` | `tabpfn-ts`, `chronos-2`   | Model to evaluate        |

Generate monthly stacked SHAP waterfall plots:

```bash
uv run src/stacked_plot.py
```

Generate SHAP waterfall plots with exogenous weather overlays (monthly, 4-day panel, and single-day plots):

```bash
uv run src/stacked_plot_with_exogenous.py
```

## License

The code in this project is licensed under the MIT License — see [LICENSE](LICENSE.MD) for details.

For data sources and their respective licensing terms (Copernicus ERA5, ENTSO-E load data), see [data/README.md](data/README.md).
