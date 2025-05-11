# Bayesian LSTM for Stock Return Prediction

This project implements a **Bayesian Long Short-Term Memory (LSTM)** framework to forecast daily stock returns for 22 U.S. companies across multiple industries. It combines the power of recurrent neural networks with Bayesian techniques for both **hyperparameter tuning** and **uncertainty quantification**.

## Features

* **Bayesian Optimization** to tune LSTM hyperparameters:

  * Hidden dimension size
  * Dropout rate
  * Learning rate
* **Monte Carlo Dropout (MC Dropout)** to generate predictive distributions and 95% confidence intervals.
* Fully automated pipeline:

  * Log return calculation
  * Sliding window sequence generation
  * Model training, evaluation, and saving
  * Metric logging and result visualization

## Data

* **Source**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance` API
* **Stocks**: 22 tickers across Tech, Energy, Finance, Healthcare, Consumer, Industrial, Utilities, and Materials.
* **Training period**: 2021-01-01 to 2023-12-31
* **Testing period**: 2024-01-01 to 2024-12-31

## Evaluation Metrics

* **MSE** (Mean Squared Error)
* **PICP** (Prediction Interval Coverage Probability)
* **MPIW** (Mean Prediction Interval Width)
* Composite score used for optimization:

```
Score = -MSE + λ * PICP - 0.1 * MPIW
```

## File Structure

```
.
├── bayes_opt_lstm.py          # Main training + optimization script
├── model.py                   # Bayesian LSTM model with dropout
├── train.py                   # LSTM training class
├── predict_mc.py              # Inference with uncertainty via MC Dropout
├── dataset.py                 # Sequence generation and DataLoader setup
├── stock_close_prices_train.csv
├── stock_close_prices_test.csv
├── outputs/
│   ├── models/                # Saved model weights and parameters
│   ├── plots/                 # Confidence interval plots per stock
│   ├── predictions/           # CSVs with prediction and uncertainty
│   ├── logs/                  # Bayesian optimization logs
│   └── summary.csv            # Combined performance across stocks
└── README.md
```

## Requirements

* Python 3.8+
* PyTorch
* pandas, NumPy, matplotlib
* bayes-opt
* yfinance

## Usage

Run the full pipeline:

```bash
python bayes_opt_lstm.py
```

All results (models, plots, logs) will be saved under the `outputs/` directory.

