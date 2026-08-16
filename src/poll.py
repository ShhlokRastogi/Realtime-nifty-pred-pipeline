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


def fetch_latest_candles_binance(ticker: str, interval: str = "1m", limit: int = 5) -> pd.DataFrame:
    """
    Fetches the latest candles from Binance public API.
    
    Returns:
        DataFrame formatted exactly like our yfinance downloads.
    """
    symbol = map_ticker_to_binance(ticker)
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Binance klines format:
    # [
    #   [
    #     1499040000000,      // Kline open time (0)
    #     "0.01634790",       // Open (1)
    #     "0.80000000",       // High (2)
    #     "0.01575800",       // Low (3)
    #     "0.01577100",       // Close (4)
    #     "148976.11427815",  // Volume (5)
    #     ...
    #   ]
    # ]
    
    rows = []
    for item in data:
        timestamp = pd.to_datetime(item[0], unit='ms')
        rows.append({
            "Date": timestamp,
            "Open": float(item[1]),
            "High": float(item[2]),
            "Low": float(item[3]),
            "Close": float(item[4]),
            "Volume": float(item[5])
        })
        
    df = pd.DataFrame(rows)
    df = df.set_index("Date")
    return df


def poll_once():
    """Fetches new prices, saves to DB, updates features, and caches to Redis."""
    print("\n=== Polling cycle started ===")
    for ticker in TICKERS:
        try:
            print(f"Fetching latest data for {ticker} from Binance...")
            # Fetch latest 15-minute candles
            df = fetch_latest_candles_binance(ticker, interval="15m", limit=5)
            
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
