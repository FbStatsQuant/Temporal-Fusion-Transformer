# Multi-Stock Return Forecasting with Temporal Fusion Transformer

This project applies a Temporal Fusion Transformer (TFT) to multi-entity daily stock return forecasting. A single probabilistic forecasting model is trained jointly on five large-cap U.S. stocks from different market sectors, using both stock-specific technical features and shared market-wide signals from the S&P 500.

The goal is not to predict prices directly, but to produce one-step-ahead probabilistic forecasts of daily log returns through quantile prediction.

## Project Overview

Daily stock returns are noisy, weakly predictable, and often close to conditionally mean-zero. Instead of producing only a point forecast, this project uses a Temporal Fusion Transformer to estimate conditional quantiles of the next-day return distribution.

The model predicts three quantiles:

```text
q = 0.025, 0.500, 0.975
```

These correspond to:

* Lower 2.5% quantile
* Median forecast
* Upper 97.5% quantile

Together, these define a 95% predictive interval for next-day log returns.

## Assets Forecasted

The project models five stocks, each from a different sector:

| Ticker | Company           | Sector      |
| ------ | ----------------- | ----------- |
| AAPL   | Apple Inc.        | Technology  |
| JPM    | JPMorgan Chase    | Financials  |
| XOM    | ExxonMobil        | Energy      |
| JNJ    | Johnson & Johnson | Healthcare  |
| UNP    | Union Pacific     | Industrials |

The S&P 500 is used as a shared market-wide covariate.

## Data

The notebook uses the following raw data files:

```text
top50_adjclose_2010_2025.csv
unp_us_d.csv
^spx_d.csv
```

The data is filtered to the common trading-day intersection across all sources.

The final modeling window is:

```text
2015-04-06 to 2025-12-30
```

The data is converted into a long-format panel with one row per stock-date pair.

Final panel size:

```text
13,510 rows = 2,702 trading days × 5 stocks
```

## Feature Engineering

The model operates on log returns rather than raw prices:

```text
r_t = log(P_t / P_{t-1})
```

For each stock, the notebook constructs:

* Daily log return
* 5-day rolling mean
* 21-day rolling mean
* 63-day rolling mean
* 5-day rolling volatility
* 21-day rolling volatility
* 63-day rolling volatility

Market-wide features:

* S&P 500 daily log return
* S&P 500 21-day rolling volatility

Known future covariates:

* Day of week
* Month

Static covariates:

* Stock sector

The first 63 rows are dropped to remove missing values from the longest rolling window.

## Train / Validation / Test Split

The data is split chronologically to avoid lookahead bias:

| Split      | Proportion |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

The model is trained only on past data and evaluated on the held-out final period.

## Model

The project uses the `TemporalFusionTransformer` implementation from `pytorch-forecasting`.

Main hyperparameters:

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Encoder length          | 63 to 126 days    |
| Prediction length       | 1 day             |
| Hidden size             | 32                |
| LSTM layers             | 1                 |
| Attention heads         | 4                 |
| Dropout                 | 0.2               |
| Hidden continuous size  | 16                |
| Batch size              | 128               |
| Learning rate           | 1e-3              |
| Loss                    | Quantile Loss     |
| Quantiles               | 0.025, 0.5, 0.975 |
| Max epochs              | 100               |
| Early stopping patience | 10                |

The model has approximately:

```text
74,431 trainable parameters
```

## Training

Training is performed using PyTorch Lightning.

The model uses early stopping based on validation loss and saves the best checkpoint to:

```text
checkpoints/tft-best.ckpt
```

In the notebook run, early stopping selected the best model around epoch 30.

Best validation loss:

```text
0.004019
```

## Evaluation

The TFT is evaluated against a simple naive benchmark:

```text
Predicted return = 0
Prediction interval = historical training volatility × ±1.96
```

The benchmark is strong because daily stock returns are difficult to forecast directionally, and zero-return forecasts are often competitive at short horizons.

### TFT Test Metrics

| Stock | Pinball 0.025 | Pinball 0.50 | Pinball 0.975 | 95% Coverage | Avg Interval Width |
| ----- | ------------: | -----------: | ------------: | -----------: | -----------------: |
| AAPL  |      0.001474 |     0.006014 |      0.001530 |       89.66% |           0.056491 |
| JPM   |      0.001432 |     0.005554 |      0.001190 |       92.36% |           0.056350 |
| XOM   |      0.001011 |     0.005479 |      0.000826 |       94.83% |           0.056236 |
| JNJ   |      0.000942 |     0.004069 |      0.000829 |       97.29% |           0.056619 |
| UNP   |      0.000999 |     0.004910 |      0.001091 |       95.07% |           0.056282 |

### Naive Benchmark Metrics

| Stock | Pinball 0.025 | Pinball 0.50 | Pinball 0.975 | 95% Coverage | Avg Interval Width |
| ----- | ------------: | -----------: | ------------: | -----------: | -----------------: |
| AAPL  |      0.001324 |     0.005903 |      0.001370 |       95.32% |           0.073280 |
| JPM   |      0.001308 |     0.005472 |      0.001169 |       96.55% |           0.070774 |
| XOM   |      0.001102 |     0.005310 |      0.000931 |       98.77% |           0.072381 |
| JNJ   |      0.000835 |     0.004042 |      0.000777 |       97.04% |           0.046212 |
| UNP   |      0.001029 |     0.004811 |      0.001110 |       97.54% |           0.065740 |

## Interpretation

The TFT does not meaningfully outperform the naive zero-return benchmark for median return prediction. This is expected: daily stock returns have very low signal-to-noise ratios, and short-horizon directional predictability is weak.

The more interesting result is uncertainty modeling.

The naive benchmark uses a fixed interval width based on historical volatility. The TFT, by contrast, produces time-varying intervals that widen during volatile periods and contract during calmer periods. This makes the model more useful as a dynamic risk forecasting tool than as a pure directional trading signal.

However, the TFT intervals are somewhat overconfident for some assets, especially AAPL, where the 95% interval coverage is below the nominal level.

## Main Takeaways

* A single TFT can jointly model multiple financial time series.
* Quantile forecasts are more informative than point forecasts for daily returns.
* Median return prediction remains difficult and does not clearly beat a zero-return baseline.
* TFT intervals adapt over time, unlike fixed-volatility naive intervals.
* The model is better interpreted as a probabilistic risk model than as a trading signal generator.

## Repository Contents

```text
.
├── 1_TFT_5_Stocks.ipynb          # Main notebook
├── tft_panel.csv                 # Generated long-format panel dataset
├── checkpoints/
│   └── tft-best.ckpt             # Best TFT checkpoint after training
├── top50_adjclose_2010_2025.csv  # Adjusted close data for selected stocks
├── unp_us_d.csv                  # Union Pacific daily data
├── ^spx_d.csv                    # S&P 500 daily index data
└── README.md
```

## Requirements

Core libraries:

```text
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
torch
lightning
pytorch-forecasting
pytorch-lightning
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels torch lightning pytorch-lightning pytorch-forecasting
```

The notebook also imports TensorFlow/Keras, but the main forecasting model is implemented with PyTorch Forecasting.

## How to Run

1. Clone the repository.

```bash
git clone <repo-url>
cd <repo-name>
```

2. Install the required packages.

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels torch lightning pytorch-lightning pytorch-forecasting
```

3. Place the required CSV files in the project directory:

```text
top50_adjclose_2010_2025.csv
unp_us_d.csv
^spx_d.csv
```

4. Open and run the notebook:

```text
1_TFT_5_Stocks.ipynb
```

5. The notebook will:

* Load and merge the raw price data
* Construct log-return and volatility features
* Save the panel dataset as `tft_panel.csv`
* Train the TFT model
* Save the best checkpoint
* Generate test-set quantile forecasts
* Compare the TFT against a naive benchmark
* Plot return forecasts, price forecasts, interval widths, and feature importance

## Possible Improvements

Potential extensions include:

* Add more stocks to improve cross-sectional learning
* Include macroeconomic or factor-based covariates
* Add realized volatility or intraday volatility features
* Compare against GARCH, HAR-RV, LSTM, and transformer baselines
* Calibrate prediction intervals using conformal prediction
* Tune TFT hyperparameters with cross-validation over rolling windows
* Evaluate trading performance after transaction costs
* Separate directional forecasting from volatility forecasting
* Use a rolling retraining scheme instead of a single static split

## Conclusion

This project demonstrates a probabilistic deep learning approach to multi-stock return forecasting using the Temporal Fusion Transformer. The model does not produce strong directional forecasts, but it provides dynamic quantile-based uncertainty estimates that adapt to changing market regimes. Its main value is therefore in risk-aware forecasting rather than direct return prediction.
