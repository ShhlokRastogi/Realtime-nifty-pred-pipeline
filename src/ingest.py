"""
Data ingestion — pull the last 7 days of 1-minute OHLCV candles
from Binance public API and save to Postgres + CSV backup.
"""
import os
import time
import requests
import pandas as pd
import psycopg2
from config import TICKERS, RAW_DATA_DIR, DB_CONFIG


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Fetches the last 7 days of 1-minute candles from Binance public API.
    """
    # Map BTC-USD to BTCUSDT
    symbol = ticker.replace("-USD", "USDT")
    url = "https://api.binance.com/api/v3/klines"
    
    # 3 years in milliseconds
    one_day_ms = 24 * 60 * 60 * 1000
    total_duration_ms = 3 * 365 * one_day_ms
    
    end_time = int(time.time() * 1000)
    start_time = end_time - total_duration_ms
    
    current_start = start_time
    all_candles = []
    
    print(f"Downloading 3 years of 15m data for {symbol}...")
    
    # Pull in batches of 1000 (Binance limit per request)
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_candles.extend(data)
        
        # Set next start time to the close time of the last candle + 1ms
        current_start = data[-1][6] + 1
        time.sleep(0.1) # Small rate-limit delay
        
    # Format into DataFrame
    rows = []
    for item in all_candles:
        rows.append({
            "Date": pd.to_datetime(item[0], unit='ms'),
            "Open": float(item[1]),
            "High": float(item[2]),
            "Low": float(item[3]),
            "Close": float(item[4]),
            "Volume": float(item[5])
        })
        
    df = pd.DataFrame(rows)
    df = df.set_index("Date")
    df = df.sort_index()
    # Remove duplicates if any overlap occurred
    df = df[~df.index.duplicated(keep='first')]
    return df


def save_to_csv(df: pd.DataFrame, ticker: str) -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path, index=True)


from psycopg2.extras import execute_values
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
# Register numpy adapters to prevent "schema np does not exist" formatting errors
register_adapter(np.float64, AsIs)
register_adapter(np.int64, AsIs)

def save_to_postgres(df: pd.DataFrame, ticker: str) -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Convert DataFrame into a list of tuples for bulk insertion
    tuples = [
        (
            ticker,
            date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
            float(rows['Open']),
            float(rows['High']),
            float(rows['Low']),
            float(rows['Close']),
            float(rows['Volume'])
        )
        for date, rows in df.iterrows()
    ]

    query = """
        INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, date) DO UPDATE
        SET open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
    """

    # Run bulk insertion in batches of 10,000 to prevent statement timeouts over WAN
    batch_size = 10000
    for i in range(0, len(tuples), batch_size):
        batch = tuples[i : i + batch_size]
        print(f"  Inserting batch {i // batch_size + 1} ({len(batch)} rows)...")
        execute_values(cursor, query, batch)
    
    conn.commit()
    cursor.close()
    conn.close()


def ingest_all() -> dict:
    data = {}
    for t in TICKERS:
        df = fetch_ohlcv(t)
        save_to_postgres(df, t)
        save_to_csv(df, t)
        print(f"{t}: saved {len(df)} 1-minute rows to CSV and Postgres")
        data[t] = df
    return data


if __name__ == "__main__":
    data = ingest_all()
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows, {df.index[0]} to {df.index[-1]}")