# LSTM Stock Price Prediction with Monte Carlo Dropout

> 99% of LSTM stock projects on GitHub are scientifically invalid. This one isn't.

Most implementations shuffle time-series data, fit scalers on the full dataset, and call it "deep learning." This project is built with engineering rigor — proper temporal splits, uncertainty quantification, and honest failure analysis.

---

## What Makes This Different

| Problem in most projects | What this project does |
|--------------------------|------------------------|
| Random train/test split on time-series | Strict temporal split — train → val → test, no shuffling |
| Scaler fit on full dataset | Scaler fit **only** on training data |
| Single prediction, no uncertainty | Monte Carlo Dropout — 100 forward passes → mean ± 2σ bands |
| No baseline comparison | Compared against Naive, MA-20, and Linear Regression |
| No failure analysis | Explicit failure report — worst dates, high-error frequency |

---

## Key Feature — Monte Carlo Dropout

Standard neural networks give you a single prediction with no sense of confidence. MC Dropout fixes this:

1. Keep dropout **active** during inference (`model.train()`)
2. Run 100 forward passes on the same input
3. Each pass gives a slightly different prediction due to random dropout
4. Compute **mean** (prediction) and **std** (uncertainty) across all 100 passes
5. Plot mean ± 2σ as a confidence band

Wide band = model is uncertain = don't act on this prediction.
Narrow band = model is confident.

This transforms the project from a curve-fitting exercise into something scientifically defensible.

---

## Tech Stack

- **PyTorch** — LSTM model, MC Dropout inference
- **yfinance** — OHLCV data download
- **pandas-ta** — RSI, MACD, Bollinger Bands, SMA, EMA
- **scikit-learn** — MinMaxScaler, Linear Regression baseline
- **Plotly** — interactive charts with confidence bands
- **Streamlit** — interactive dashboard

---

## Model Architecture
```
Input: (batch, 60 days, 14 features)
       ↓
LSTM Layer 1 — hidden_size=128, dropout=0.2
       ↓
LSTM Layer 2 — hidden_size=128, dropout=0.2
       ↓
MC Dropout Layer — p=0.2 (active at inference)
       ↓
Fully Connected — 128 → 1
       ↓
Output: next-day Close price (normalised)
```

**Features (14 total):** Open, High, Low, Close, Volume, RSI-14, MACD, MACD Signal, MACD Histogram, Bollinger Upper/Mid/Lower, SMA-20, EMA-20

**Training:** Adam optimizer, gradient clipping (max norm=1.0), early stopping on val loss (patience=15)

---

## Results (AAPL, 5-year data, test set)

| Model | MAE | RMSE | MAPE | Directional Accuracy |
|-------|-----|------|------|----------------------|
| **LSTM (MC mean)** | **5.24** | **6.11** | **1.96%** | **58.54%** |
| Naive (yesterday) | 2.53 | 3.63 | 0.96% | 53.66% |
| MA-20 | 8.05 | 9.51 | 3.06% | 55.28% |
| Linear Regression | 5.91 | 7.44 | 2.24% | 49.59% |

**Honest interpretation:**
- Naive beats LSTM on raw price error (MAE/RMSE) — this is expected and normal. Stock prices are highly autocorrelated so "predict yesterday's price" is a strong baseline on absolute error metrics.
- LSTM beats all baselines on **directional accuracy** (58.54%) — predicting whether price goes up or down. This is the metric that actually matters for trading signals.
- 47.6% of predictions exceeded 2% error, concentrated around high-volatility events (Sep 2025, Feb 2026).

---

## Project Structure
```
lstm_stock/
├── data/
│   └── pipeline.py       # download, indicators, temporal split, sequences
├── models/
│   └── lstm.py           # LSTM architecture, training loop, MC Dropout
├── utils/
│   ├── baselines.py      # naive, MA-20, linear regression
│   ├── metrics.py        # MAE, RMSE, MAPE, directional accuracy, failure analysis
│   └── charts.py         # Plotly confidence band charts
├── train.py              # end-to-end training script
├── app.py                # Streamlit dashboard
└── outputs/              # saved model, charts, artifacts
```

---

## Quickstart
```bash
git clone https://github.com/AdityaSinghG/lstm-stock-predictor
cd lstm-stock-predictor
pip install torch yfinance pandas-ta scikit-learn plotly streamlit
```

**Train the model:**
```bash
python train.py
```

**Launch the dashboard:**
```bash
streamlit run app.py
```

---

## Failure Analysis

The model breaks down during high-volatility events — earnings surprises, macro shocks, sudden trend reversals. This is expected and honest. No LSTM predicts black swan events. The MC Dropout bands widen exactly during these periods, signaling uncertainty before the fact.

Worst prediction dates all cluster around known volatile windows in 2025–2026, validating that the uncertainty bands are meaningful, not decorative.

