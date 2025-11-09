# Crypto TDA Bot 4 — On‑the‑fly N‑BEATS Backtesting and Evaluation

A research repo for crypto time‑series forecasting and strategy backtesting that combines Topological Data Analysis (TDA) features with an N‑BEATS neural network, trained on‑the‑fly for every test window. The project includes tools to statistically evaluate performance across many randomized periods and hyperparameter grids.

## What’s inside

- Data acquisition from Binance (via `ccxt`)
- TDA feature engineering (via `giotto-tda`)
- N‑BEATS model in PyTorch (`nbeats_model.py`)
- Backtesting with on‑the‑fly train/test split and multiple strategies
  - Long/short
  - Short‑only
  - Buy & Hold baseline
  - Moving‑average crossover baseline
- Evaluation utilities
  - Randomized multi‑run evaluation (`evaluation.py`)
  - Grid search across parameter sets with plots
  - Focused evaluation of the “best” parameters found (`evaluation_focused.py`)

## Repository layout

- `data_acquisition.py` — Fetch OHLCV market data (Binance via `ccxt`)
- `tda_feature_engineering.py` — Build TDA feature matrices from price series
- `nbeats_model.py` — Shared N‑BEATS model definition (PyTorch)
- `nbeats_training.py` — Standalone training script for persistent models (optional)
- `backtesting.py` — End‑to‑end pipeline: fetch → engineer features → train N‑BEATS → predict → backtest strategies
- `evaluation.py` — Run many backtests over randomized periods and parameter grids; summarize and plot results
- `evaluation_focused.py` — Targeted experiments around a promising parameter set
- `README.md` — You’re here

Note: Some plots and results are generated into run‑specific folders (e.g., `evaluation_results/`, `evaluation_focused_results/`). Consider ignoring these in Git to keep the repo clean.

## Getting started

### Prerequisites

- Python 3.10+ recommended
- Windows PowerShell (commands below use PowerShell syntax)

### Create and activate a virtual environment

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can temporarily allow it for this session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Install dependencies

If you have a `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If not, install the core packages manually:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision torchaudio ccxt giotto-tda
```

Note: `giotto-tda` may require build tools; refer to its docs if installation fails.

## Usage

### 1) Run a single backtest

`backtesting.py` trains the N‑BEATS model on‑the‑fly for the chosen window, then evaluates one or more strategies on a test segment. Typical usage:

```powershell
python backtesting.py --symbol BTC/USDT --timeframe 1h --window_size 60 --forecast_intervals 1 --epochs 50 --strategy longshort --json_output
```

Tips:
- Omit `--json_output` to see plots/equity curves interactively.
- Use `--strategy shortonly` to run the short‑only variant, or `--strategy macrossover` / `--strategy buyhold` for baselines.
- Run `python backtesting.py --help` to see all available options and defaults in your version.

### 2) Evaluate statistically (many randomized runs)

`evaluation.py` runs multiple backtests over randomized date ranges and/or a grid of parameters, then aggregates results and renders summaries/plots.

```powershell
python evaluation.py
```

Outputs typically include:
- A CSV/JSON summary of final equity per run and per parameter set
- A seaborn box plot comparing parameter sets
- Optional histograms of final equity distribution

Adjust the `param_grid` in `evaluation.py` to try different window sizes, horizons, epochs, etc.

### 3) Focused evaluation around a “winner”

`evaluation_focused.py` is a clone of the evaluation tool configured to explore variations near the best‑performing parameters found earlier (e.g., `window_size=60`, `forecast_intervals=1`).

```powershell
python evaluation_focused.py
```

This keeps previous results intact and reproducible.

## How it works (high level)

1. Data: Fetch OHLCV using `ccxt` from Binance.
2. Features: Build TDA features via sliding windows and persistence diagrams (`giotto-tda`).
3. Model: Train an N‑BEATS model (`torch`) on the training split of your selected window.
4. Predict: Generate forecasts on the test split.
5. Backtest: Convert forecasts into trade signals per strategy; compute equity curves and summary stats.
6. Evaluate: Repeat across randomized periods and hyperparameter grids; summarize and plot.

## Notes and caveats

- Research code: This is a research sandbox. Expect iteration and change.
- Reproducibility: Results vary by random seeds, market regimes, and data ranges. Use the evaluation scripts to quantify variability.
- No financial advice: This code is for research and education only. Trading involves substantial risk.

## Contributing / branching

Active development for on‑the‑fly training is on branch `onTheFly`. Use feature branches and pull requests for changes.

## License

If this repository lacks a license file, default copyright applies. Consider adding an open‑source license (e.g., MIT) if you plan to share/extend.
