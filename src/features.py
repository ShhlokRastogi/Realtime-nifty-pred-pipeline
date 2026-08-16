import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from config import (
    RSI_PERIOD, SMA_SHORT, SMA_LONG,
    VOLATILITY_WINDOW, VOLUME_DELTA_WINDOW,
    LAGGED_RETURN_PERIODS
)

def lagged_returns(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """
    Calculate lagged returns for specified periods.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' prices.
    periods (list): List of periods for which to calculate lagged returns.

    Returns:
    pd.DataFrame: DataFrame with lagged return columns added.
    """
    for period in periods:
        df[f'lagged_return_{period}'] = df['close'].pct_change(periods=period)
    return df

def sma_crossover(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA) crossover signals.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' prices.
    short_window (int): Window size for the short-term SMA.
    long_window (int): Window size for the long-term SMA.

    Returns:
    pd.DataFrame: DataFrame with SMA crossover signals added.
    """
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    df['sma_crossover'] = df['sma_short'] - df['sma_long']
    return df

def rolling_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' prices.
    window (int): Window size for calculating rolling volatility.

    Returns:
    pd.DataFrame: DataFrame with rolling volatility added.
    """
    df['returns'] = df['close'].pct_change()
    df['rolling_volatility'] = df['returns'].rolling(window=window).std()
    return df

def volume_delta(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate volume delta (change in volume over a specified window).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'volume' data.
    window (int): Window size for calculating volume delta.

    Returns:
    pd.DataFrame: DataFrame with volume delta added.
    """
    vol_mean= df['volume'].rolling(window=window).mean()
    df['volume_delta'] = (df['volume'] - vol_mean)/vol_mean
    return df

def rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' prices.
    period (int): Period for calculating RSI.

    Returns:
    pd.DataFrame: DataFrame with RSI added.
    """
    rsi_indicator = RSIIndicator(close=df['close'], window=period)
    df['rsi'] = rsi_indicator.rsi()
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    df = lagged_returns(df, LAGGED_RETURN_PERIODS)
    df = sma_crossover(df, SMA_SHORT, SMA_LONG)
    df = rolling_volatility(df, VOLATILITY_WINDOW)
    df = volume_delta(df, VOLUME_DELTA_WINDOW)
    df = rsi(df, RSI_PERIOD)
    df = df.drop(columns=['sma_short', 'sma_long', 'returns'])  # intermediate cols
    df = df.dropna()
    return df

if __name__ == "__main__":
    data = pd.read_csv("data/raw/BTC-USD.csv", index_col=0, parse_dates=True)
    df = build_features(data)
    print(f"Features built: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")