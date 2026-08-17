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


import requests

def fetch_latest_candles_coinbase(ticker: str, limit: int = 5) -> pd.DataFrame:
    """
    Fetches the latest 15-minute candles from the Coinbase public exchange API.
    Bypasses US IP blocks and is not rate-limited on Render.
    
    Returns:
        DataFrame formatted exactly like our downloads.
    """
    print(f"  Downloading latest 15m candles from Coinbase...")
    
    # Coinbase API uses the format BTC-USD directly
    url = f"https://api.exchange.coinbase.com/products/{ticker}/candles"
    params = {"granularity": 900} # 15 minutes = 900 seconds
    headers = {"User-Agent": "Mozilla/5.0"} # Coinbase requires a User-Agent header
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json() # List of lists: [time, low, high, open, close, volume]
    
    rows = []
    # Take the latest N candles (Coinbase returns them descending, so we limit first)
    for item in data[:limit]:
        rows.append({
            "Date": pd.to_datetime(item[0], unit='s'),
            "Low": float(item[1]),
            "High": float(item[2]),
            "Open": float(item[3]),
            "Close": float(item[4]),
            "Volume": float(item[5])
        })
        
    df = pd.DataFrame(rows)
    df = df.set_index("Date")
    df = df.sort_index() # Sort ascending
    return df


def poll_once():
    """Fetches new prices, saves to DB, updates features, and caches to Redis."""
    print("\n=== Polling cycle started ===")
    for ticker in TICKERS:
        try:
            print(f"Fetching latest data for {ticker} from Coinbase...")
            # Fetch latest 15-minute candles from Coinbase
            df = fetch_latest_candles_coinbase(ticker, limit=5)
            
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
