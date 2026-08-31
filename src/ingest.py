import os
import yfinance as yf
import pandas as pd
from config import TICKERS, HIST_START, HIST_END, RAW_DATA_DIR

def ingest_historical_data():
    """Downloads 2 years of hourly candles from Yahoo Finance and saves to raw data dir."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    print(f"Ingesting hourly data for tickers: {TICKERS}...")
    
    # Download raw data
    df_nifty = yf.download("^NSEI", period="730d", interval="1h")
    df_vix = yf.download("^INDIAVIX", period="730d", interval="1h")
    
    # Clean multi-level columns if present
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    # Convert timezone to Asia/Kolkata first, then make timezone-naive
    if df_nifty.index.tz is not None:
        df_nifty.index = df_nifty.index.tz_convert("Asia/Kolkata")
    df_nifty.index = df_nifty.index.tz_localize(None)
    
    if df_vix.index.tz is not None:
        df_vix.index = df_vix.index.tz_convert("Asia/Kolkata")
    df_vix.index = df_vix.index.tz_localize(None)
    
    # Save raw CSVs
    nifty_path = os.path.join(RAW_DATA_DIR, "nifty_raw.csv")
    vix_path = os.path.join(RAW_DATA_DIR, "vix_raw.csv")
    
    df_nifty.to_csv(nifty_path)
    df_vix.to_csv(vix_path)
    
    print(f"Saved Nifty 50 raw to: {nifty_path}")
    print(f"Saved India VIX raw to: {vix_path}")

if __name__ == "__main__":
    ingest_historical_data()