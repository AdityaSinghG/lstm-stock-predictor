import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    period: str = "5y"
    seq_len: int = 60
    train_frac: float = 0.70
    val_frac: float = 0.15


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
    "SMA_20", "EMA_20",
]
TARGET_COL = "Close"


def download_data(cfg: DataConfig) -> pd.DataFrame:
    df = yf.download(cfg.ticker, period=cfg.period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"[data] Raw rows: {len(df)}")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.dropna(inplace=True)
    print(f"[data] After indicators: {len(df)} rows")
    return df


def temporal_split(df: pd.DataFrame, cfg: DataConfig):
    """Hard cutoff — NO shuffling, NO leakage."""
    n = len(df)
    train_end = int(n * cfg.train_frac)
    val_end   = int(n * (cfg.train_frac + cfg.val_frac))
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]
    print(f"[data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_scaler(train_df: pd.DataFrame):
    """Fit ONLY on train. Never touch val/test before transform."""
    available = [c for c in FEATURE_COLS if c in train_df.columns]
    scaler = MinMaxScaler()
    scaler.fit(train_df[available])
    return scaler, available


def make_sequences(df, scaler, available_cols, seq_len) -> Tuple[np.ndarray, np.ndarray]:
    scaled = scaler.transform(df[available_cols])
    target_idx = available_cols.index(TARGET_COL)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_close(values, scaler, available_cols):
    target_idx = available_cols.index(TARGET_COL)
    dummy = np.zeros((len(values), len(available_cols)), dtype=np.float32)
    dummy[:, target_idx] = values
    return scaler.inverse_transform(dummy)[:, target_idx]


def load_pipeline(cfg: DataConfig):
    df       = download_data(cfg)
    df       = add_indicators(df)
    train_df, val_df, test_df = temporal_split(df, cfg)
    scaler, available_cols    = build_scaler(train_df)

    X_train, y_train = make_sequences(train_df, scaler, available_cols, cfg.seq_len)
    X_val,   y_val   = make_sequences(val_df,   scaler, available_cols, cfg.seq_len)
    X_test,  y_test  = make_sequences(test_df,  scaler, available_cols, cfg.seq_len)

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        test_close=test_df[TARGET_COL].values[cfg.seq_len:],
        test_dates=test_df.index[cfg.seq_len:],
        scaler=scaler,
        available_cols=available_cols,
        df=df,
        train_df=train_df, val_df=val_df, test_df=test_df,
    )