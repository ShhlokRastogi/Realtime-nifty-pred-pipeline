import os
import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from features import calculate_technical_features, apply_fractional_differentiation

def build_and_store_features():
    """Reads raw CSVs, merges them, generates the feature set, and saves to processed folder."""
    nifty_path = os.path.join(RAW_DATA_DIR, "nifty_raw.csv")
    vix_path = os.path.join(RAW_DATA_DIR, "vix_raw.csv")
    
    if not os.path.exists(nifty_path) or not os.path.exists(vix_path):
        raise FileNotFoundError("Raw files missing. Please run ingest.py first.")
        
    df_nifty = pd.read_csv(nifty_path, index_col='Datetime', parse_dates=True)
    df_vix = pd.read_csv(vix_path, index_col='Datetime', parse_dates=True)
    
    # Merge datasets
    df_vix_close = df_vix[['Close']].rename(columns={'Close': 'vix'})
    df_merged = df_nifty.join(df_vix_close, how='inner')
    df_merged['vix'] = df_merged['vix'].ffill().bfill()
    df_merged['vix_return'] = df_merged['vix'].pct_change(1).fillna(0.0)
    
    df_merged = df_merged.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    # Calculate Features
    df_features = calculate_technical_features(df_merged)
    df_features = apply_fractional_differentiation(df_features)
    
    # Save processed features
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_features.csv")
    df_features.to_csv(processed_path)
    
    print(f"Processed feature matrix successfully saved to: {processed_path}")

if __name__ == "__main__":
    build_and_store_features()