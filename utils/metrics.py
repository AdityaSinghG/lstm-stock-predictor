import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error — interpretable as % off."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Did the model predict the RIGHT DIRECTION (up/down)?
    This is the metric that actually matters for trading.
    """
    actual_dir    = np.diff(actual) > 0
    predicted_dir = np.diff(predicted) > 0
    return np.mean(actual_dir == predicted_dir) * 100


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    mae   = mean_absolute_error(actual, predicted)
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mape_ = mape(actual, predicted)
    da    = directional_accuracy(actual, predicted)

    return {
        "model": name,
        "MAE":   round(mae,   4),
        "RMSE":  round(rmse,  4),
        "MAPE":  round(mape_, 4),
        "DirectionalAccuracy": round(da, 2),
    }


def failure_analysis(actual: np.ndarray, predicted: np.ndarray,
                     dates, threshold_pct: float = 2.0) -> dict:
    """
    Honest failure analysis — find where the model breaks down.
    
    threshold_pct: flag predictions where error > X% of actual price.
    This is the section that makes your project scientifically credible.
    """
    errors     = np.abs(actual - predicted)
    pct_errors = (errors / actual) * 100
    
    bad_mask   = pct_errors > threshold_pct
    bad_dates  = dates[bad_mask]
    bad_errors = pct_errors[bad_mask]

    # Worst 5 predictions
    worst_idx  = np.argsort(pct_errors)[-5:][::-1]

    return {
        "total_predictions":   len(actual),
        "high_error_count":    int(bad_mask.sum()),
        "high_error_pct":      round(float(bad_mask.mean()) * 100, 2),
        "mean_error_on_bad":   round(float(bad_errors.mean()), 4) if bad_mask.any() else 0,
        "worst_dates":         [str(dates[i])[:10] for i in worst_idx],
        "worst_pct_errors":    [round(pct_errors[i], 2) for i in worst_idx],
        "threshold_used":      threshold_pct,
    }


def print_comparison_table(metrics_list: list[dict]):
    """Pretty print all models side by side."""
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'DirAcc%':>10}"
    print("\n" + "=" * 58)
    print(header)
    print("-" * 58)
    for m in metrics_list:
        print(f"{m['model']:<20} {m['MAE']:>8} {m['RMSE']:>8} "
              f"{m['MAPE']:>8} {m['DirectionalAccuracy']:>10}")
    print("=" * 58 + "\n")


def print_failure_report(report: dict):
    print("\n--- Failure Analysis ---")
    print(f"Total predictions : {report['total_predictions']}")
    print(f"High error (>{report['threshold_used']}%) : "
          f"{report['high_error_count']} ({report['high_error_pct']}% of predictions)")
    print(f"Mean error on bad : {report['mean_error_on_bad']}%")
    print("\nWorst 5 predictions:")
    for d, e in zip(report['worst_dates'], report['worst_pct_errors']):
        print(f"  {d}  →  {e}% error")
    print()