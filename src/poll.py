"""
Live Polling Script — queries public Binance API for the latest candles,
saves them to Postgres, and runs the feature store to update Redis.

Run: python src/poll.py
"""
import time
import requests
import pandas as pd
from config import TICKERS
from ingest import save_to_postgres
from feature_store import run_feature_store


def map_ticker_to_binance(ticker: str) -> str:
    """Maps tickers like BTC-USD to Binance symbols like BTCUSDT."""
    clean = ticker.replace("-USD", "USDT")
    return clean


import yfinance as yf

def fetch_latest_candles_yfinance(ticker: str, interval: str = "15m", limit: int = 5) -> pd.DataFrame:
    """
    Fetches the latest candles from Yahoo Finance, bypassing US IP blocks.
    
    Returns:
        DataFrame formatted exactly like our yfinance downloads.
    """
    print(f"  Downloading latest {interval} candles from Yahoo Finance...")
    # Fetch the last 1 day of 15m candles
    df = yf.download(tickers=ticker, period="1d", interval=interval, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")
        
    # If columns are MultiIndexed (happens in newer yfinance versions), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Sort and keep the last N candles
    df = df.sort_index().tail(limit)
    return df


def poll_once():
    """Fetches new prices, saves to DB, updates features, and caches to Redis."""
    print("\n=== Polling cycle started ===")
    for ticker in TICKERS:
        try:
            print(f"Fetching latest data for {ticker} from Yahoo Finance...")
            # Fetch latest 15-minute candles from Yahoo Finance
            df = fetch_latest_candles_yfinance(ticker, interval="15m", limit=5)
            
            # Upsert into Postgres ohlcv table
            save_to_postgres(df, ticker)
            print(f"  Successfully saved/updated {len(df)} rows in database.")
            
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")
            
    # Step 2: Trigger feature store to recompute and write to Postgres + Redis
    try:
        print("Triggering Feature Store run (incremental mode: limit=100)...")
        run_feature_store(limit=100)
    except Exception as e:
        print(f"  Error during Feature Store run: {e}")
    print("=== Polling cycle complete ===")


def main():
    print("Starting Live Polling Service (Interval: 60s)...")
    while True:
        poll_once()
        time.sleep(60)


if __name__ == "__main__":
    main()
