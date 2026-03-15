import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_predictions(dates, actual, mean_pred, std_pred,
                     naive=None, ma=None, lr=None,
                     ticker="STOCK", n_std=2):
    """
    Main prediction chart with MC Dropout confidence bands
    and all 3 baseline comparisons.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Predictions vs Actual", "Prediction Error (%)"),
        vertical_spacing=0.08,
    )

    upper = mean_pred + n_std * std_pred
    lower = mean_pred - n_std * std_pred

    # Confidence band (shaded)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"±{n_std}σ Confidence Band",
        showlegend=True,
    ), row=1, col=1)

    # Actual price
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        line=dict(color="#ffffff", width=2),
        name="Actual",
    ), row=1, col=1)

    # LSTM mean prediction
    fig.add_trace(go.Scatter(
        x=dates, y=mean_pred,
        line=dict(color="#636EFA", width=2),
        name="LSTM (MC mean)",
    ), row=1, col=1)

    # Baselines
    if naive is not None:
        fig.add_trace(go.Scatter(
            x=dates[-len(naive):], y=naive,
            line=dict(color="#EF553B", width=1, dash="dash"),
            name="Naive",
        ), row=1, col=1)

    if ma is not None:
        fig.add_trace(go.Scatter(
            x=dates[-len(ma):], y=ma,
            line=dict(color="#00CC96", width=1, dash="dash"),
            name="MA-20",
        ), row=1, col=1)

    if lr is not None:
        fig.add_trace(go.Scatter(
            x=dates[-len(lr):], y=lr,
            line=dict(color="#FFA15A", width=1, dash="dash"),
            name="Linear Reg",
        ), row=1, col=1)

    # Error % subplot
    pct_error = np.abs(actual - mean_pred) / actual * 100
    fig.add_trace(go.Scatter(
        x=dates, y=pct_error,
        fill="tozeroy",
        fillcolor="rgba(239, 85, 59, 0.2)",
        line=dict(color="#EF553B", width=1),
        name="Abs Error %",
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — LSTM + MC Dropout vs Baselines (Test Set)",
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Error %", row=2, col=1)

    return fig


def plot_training_history(history: dict):
    """Loss curves — sanity check that training went well."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=history["train_loss"],
        line=dict(color="#636EFA", width=2),
        name="Train Loss",
    ))
    fig.add_trace(go.Scatter(
        y=history["val_loss"],
        line=dict(color="#EF553B", width=2),
        name="Val Loss",
    ))

    fig.update_layout(
        title="Training Loss Curves",
        template="plotly_dark",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        height=400,
    )
    return fig


def plot_uncertainty_distribution(std_pred: np.ndarray, ticker="STOCK"):
    """
    Histogram of uncertainty across test period.
    High std on certain dates = model is unsure = don't trade those days.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=std_pred,
        nbinsx=40,
        marker_color="#636EFA",
        opacity=0.75,
        name="MC Dropout Std",
    ))

    fig.add_vline(
        x=np.mean(std_pred),
        line_dash="dash",
        line_color="#EF553B",
        annotation_text=f"Mean σ = {np.mean(std_pred):.4f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{ticker} — Uncertainty Distribution (MC Dropout Std)",
        template="plotly_dark",
        xaxis_title="Std Dev (normalised scale)",
        yaxis_title="Count",
        height=400,
    )
    return fig