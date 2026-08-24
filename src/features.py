import numpy as np
import pandas as pd
from config import FFD_D, FFD_MAX_LAGS

def calculate_technical_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Calculates all 16 technical, temporal, and volatility features."""
    df = df_merged.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD Diff Pct
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff_pct'] = (macd - signal) / (close + 1e-9)
    
    # Volatility Indicators
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(window=14).mean() / (close + 1e-9)
    df['hl_spread'] = (high - low) / (close + 1e-9)
    df['volume_delta'] = volume.diff() / (volume.shift(1) + 1e-9)
    df['lagged_return_1'] = close.pct_change(1)
    
    # Realized Volatility Lags
    df['realized_vol_5'] = df['lagged_return_1'].rolling(5).std()
    df['realized_vol_10'] = df['lagged_return_1'].rolling(10).std()
    df['realized_vol_20'] = df['lagged_return_1'].rolling(20).std()
    
    # Time Cyclical Features
    hours = df.index.hour
    days = df.index.dayofweek
    df['sin_hour'] = np.sin(2 * np.pi * hours / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hours / 24.0)
    df['sin_day'] = np.sin(2 * np.pi * days / 7.0)
    df['cos_day'] = np.cos(2 * np.pi * days / 7.0)
    
    return df.dropna()

def get_weights_ffd(d, size):
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    return np.array(w)

def apply_fractional_differentiation(df, d=FFD_D, max_lags=FFD_MAX_LAGS):
    series = df['close']
    w = get_weights_ffd(d, max_lags)
    w_rev = w[::-1]
    res = []
    for i in range(max_lags - 1, len(series)):
        res.append(np.dot(series.iloc[i - max_lags + 1 : i + 1], w_rev))
    df_res = df.iloc[max_lags - 1 :].copy()
    df_res['close_fracdiff'] = res
    return df_res