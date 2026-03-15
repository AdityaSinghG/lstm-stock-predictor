import streamlit as st
import torch
import pickle
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import load_pipeline, DataConfig, inverse_close
from models.lstm import LSTMModel, mc_dropout_predict
from utils.baselines import align_baselines
from utils.metrics import compute_all_metrics, failure_analysis
from utils.charts import plot_predictions, plot_training_history, plot_uncertainty_distribution

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 LSTM Stock Price Prediction")
st.caption("Monte Carlo Dropout uncertainty bands · 3 baseline comparisons · Honest failure analysis")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    ticker   = st.text_input("Ticker", value="AAPL").upper()
    period   = st.selectbox("History", ["2y", "3y", "5y"], index=2)
    seq_len  = st.slider("Sequence Length (days)", 30, 120, 60)
    mc_passes = st.slider("MC Dropout Passes", 20, 200, 100)
    run = st.button("🚀 Run Prediction", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────
if not run:
    st.info("Configure settings in the sidebar and click **Run Prediction** to start.")
    st.stop()

with st.spinner("Downloading data and adding indicators..."):
    cfg = DataConfig(ticker=ticker, period=period, seq_len=seq_len)
    try:
        data = load_pipeline(cfg)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

with st.spinner("Training LSTM model..."):
    from models.lstm import train_model
    model, history = train_model(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        epochs=150, batch_size=64, lr=1e-3, patience=15,
        device="cpu",
    )

with st.spinner(f"Running {mc_passes} MC Dropout passes..."):
    mean_norm, std_norm = mc_dropout_predict(model, data["X_test"],
                                             n_passes=mc_passes, device="cpu")
    mean_price = inverse_close(mean_norm, data["scaler"], data["available_cols"])
    std_price  = inverse_close(std_norm,  data["scaler"], data["available_cols"])
    actual     = data["test_close"]
    dates      = data["test_dates"]

# Baselines
full_test_close = data["test_df"]["Close"].values
bl_actual, naive, ma, lr = align_baselines(full_test_close, seq_len)
min_len    = min(len(mean_price), len(naive), len(ma), len(lr))
mean_price = mean_price[-min_len:]
std_price  = std_price[-min_len:]
actual     = actual[-min_len:]
dates      = dates[-min_len:]
naive      = naive[-min_len:]
ma         = ma[-min_len:]
lr         = lr[-min_len:]

# ── Metrics Row ───────────────────────────────────────────────
st.subheader("Model Performance")
metrics = [
    compute_all_metrics(actual, mean_price, "LSTM"),
    compute_all_metrics(actual, naive,       "Naive"),
    compute_all_metrics(actual, ma,          "MA-20"),
    compute_all_metrics(actual, lr,          "Linear Reg"),
]

cols = st.columns(4)
labels = ["LSTM (MC mean)", "Naive", "MA-20", "Linear Reg"]
colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]
for col, m, color in zip(cols, metrics, colors):
    with col:
        st.markdown(f"**{m['model']}**")
        st.metric("MAE",  m["MAE"])
        st.metric("MAPE", f"{m['MAPE']}%")
        st.metric("Dir. Accuracy", f"{m['DirectionalAccuracy']}%")

# ── Charts ────────────────────────────────────────────────────
st.subheader("Predictions vs Actual")
fig1 = plot_predictions(dates, actual, mean_price, std_price,
                        naive=naive, ma=ma, lr=lr, ticker=ticker)
st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Loss")
    st.plotly_chart(plot_training_history(history), use_container_width=True)
with col2:
    st.subheader("Uncertainty Distribution")
    st.plotly_chart(plot_uncertainty_distribution(std_price, ticker), use_container_width=True)

# ── Failure Analysis ──────────────────────────────────────────
st.subheader("Failure Analysis")
report = failure_analysis(actual, mean_price, dates, threshold_pct=2.0)

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", report["total_predictions"])
col2.metric("High Error Predictions", f"{report['high_error_pct']}%")
col3.metric("Mean Error on Bad Days", f"{report['mean_error_on_bad']}%")

st.markdown("**Worst 5 Predictions**")
for date, err in zip(report["worst_dates"], report["worst_pct_errors"]):
    st.markdown(f"- `{date}` → **{err}%** error")