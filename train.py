import numpy as np
import torch
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import load_pipeline, DataConfig, inverse_close
from models.lstm import train_model, mc_dropout_predict
from utils.baselines import align_baselines
from utils.metrics import compute_all_metrics, print_comparison_table, failure_analysis, print_failure_report
from utils.charts import plot_predictions, plot_training_history, plot_uncertainty_distribution


def main():
    # ── Config ────────────────────────────────────────────────
    cfg = DataConfig(
        ticker="AAPL",
        period="5y",
        seq_len=60,
        train_frac=0.70,
        val_frac=0.15,
    )
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    MC_PASSES   = 100
    SAVE_DIR    = "outputs"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[train] Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────
    data = load_pipeline(cfg)

    # ── Train ─────────────────────────────────────────────────
    model, history = train_model(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        epochs=150,
        batch_size=64,
        lr=1e-3,
        patience=15,
        device=DEVICE,
    )

    # ── MC Dropout Inference ──────────────────────────────────
    print(f"\n[inference] Running {MC_PASSES} MC Dropout passes...")
    mean_norm, std_norm = mc_dropout_predict(
        model, data["X_test"], n_passes=MC_PASSES, device=DEVICE
    )

    # Inverse transform back to price scale
    mean_price = inverse_close(mean_norm, data["scaler"], data["available_cols"])
    std_price  = inverse_close(std_norm,  data["scaler"], data["available_cols"])
    actual     = data["test_close"]
    dates      = data["test_dates"]

    # ── Baselines ─────────────────────────────────────────────
    # Use full test_df Close for baseline computation (no seq_len offset)
    full_test_close = data["test_df"]["Close"].values
    bl_actual, naive, ma, lr = align_baselines(full_test_close, cfg.seq_len)

    # Align LSTM predictions to same length as baselines
    min_len    = min(len(mean_price), len(naive), len(ma), len(lr))
    mean_price = mean_price[-min_len:]
    std_price  = std_price[-min_len:]
    actual     = actual[-min_len:]
    dates      = dates[-min_len:]
    naive      = naive[-min_len:]
    ma         = ma[-min_len:]
    lr         = lr[-min_len:]

    # ── Metrics ───────────────────────────────────────────────
    metrics = [
        compute_all_metrics(actual, mean_price, "LSTM (MC mean)"),
        compute_all_metrics(actual, naive,       "Naive"),
        compute_all_metrics(actual, ma,          "MA-20"),
        compute_all_metrics(actual, lr,          "Linear Reg"),
    ]
    print_comparison_table(metrics)

    # ── Failure Analysis ──────────────────────────────────────
    report = failure_analysis(actual, mean_price, dates, threshold_pct=2.0)
    print_failure_report(report)

    # ── Charts ────────────────────────────────────────────────
    fig1 = plot_predictions(dates, actual, mean_price, std_price,
                            naive=naive, ma=ma, lr=lr, ticker=cfg.ticker)
    fig2 = plot_training_history(history)
    fig3 = plot_uncertainty_distribution(std_price, ticker=cfg.ticker)

    fig1.write_html(f"{SAVE_DIR}/predictions.html")
    fig2.write_html(f"{SAVE_DIR}/training_loss.html")
    fig3.write_html(f"{SAVE_DIR}/uncertainty.html")
    print(f"[charts] Saved to {SAVE_DIR}/")

    # ── Save model + artifacts ────────────────────────────────
    torch.save(model.state_dict(), f"{SAVE_DIR}/model.pt")
    with open(f"{SAVE_DIR}/artifacts.pkl", "wb") as f:
        pickle.dump({
            "scaler":         data["scaler"],
            "available_cols": data["available_cols"],
            "cfg":            cfg,
            "metrics":        metrics,
            "failure_report": report,
        }, f)
    print("[save] model.pt and artifacts.pkl saved.")


if __name__ == "__main__":
    main()