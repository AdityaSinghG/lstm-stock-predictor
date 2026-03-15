import numpy as np
from sklearn.linear_model import LinearRegression


def naive_baseline(close_prices: np.ndarray) -> np.ndarray:
    """
    Predict today's close = yesterday's close.
    Shifted by 1 — the simplest possible baseline.
    """
    return close_prices[:-1]


def moving_average_baseline(close_prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    20-day MA prediction. For each day, predict using the mean
    of the previous `window` days.
    """
    preds = []
    for i in range(window, len(close_prices)):
        preds.append(close_prices[i - window:i].mean())
    return np.array(preds)


def linear_regression_baseline(close_prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Fit a linear regression on the previous `window` days,
    predict the next day. Walk-forward — no leakage.
    """
    preds = []
    for i in range(window, len(close_prices)):
        x = np.arange(window).reshape(-1, 1)
        y = close_prices[i - window:i]
        model = LinearRegression().fit(x, y)
        preds.append(model.predict([[window]])[0])
    return np.array(preds)


def align_baselines(test_close: np.ndarray, seq_len: int = 60):
    """
    All baselines need the full test_close (without seq_len offset).
    Returns aligned actuals + all 3 baseline predictions at the same length.
    
    The LSTM already outputs seq_len-offset predictions, so we align
    everything to the shortest array for fair comparison.
    """
    naive  = naive_baseline(test_close)
    ma     = moving_average_baseline(test_close, window=20)
    lr     = linear_regression_baseline(test_close, window=20)

    # Find minimum length and trim all to match
    min_len = min(len(naive), len(ma), len(lr))
    naive   = naive[-min_len:]
    ma      = ma[-min_len:]
    lr      = lr[-min_len:]
    actual  = test_close[-min_len:]

    return actual, naive, ma, lr