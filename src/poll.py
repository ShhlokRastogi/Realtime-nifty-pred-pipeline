import os
import time
import pickle
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import datetime
import redis
import json
from sklearn.preprocessing import StandardScaler
from config import SEQ_LEN, FFD_D, FFD_MAX_LAGS, DB_CONFIG, REDIS_CONFIG

import pandas_market_calendars as mcal

def merge_nifty_vix(df_nifty, df_vix):
    """Joins Nifty and VIX hourly data using left join and forward-fill only, dropping initial NaNs."""
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    if df_nifty.index.tz is not None:
        df_nifty.index = df_nifty.index.tz_convert("Asia/Kolkata")
    df_nifty.index = df_nifty.index.tz_localize(None)

    if df_vix.index.tz is not None:
        df_vix.index = df_vix.index.tz_convert("Asia/Kolkata")
    df_vix.index = df_vix.index.tz_localize(None)

    df_vix_close = df_vix[['Close']].rename(columns={'Close': 'vix'})
    df_merged = df_nifty.join(df_vix_close, how='left')
    df_merged['vix'] = df_merged['vix'].ffill()
    df_merged = df_merged.dropna(subset=['vix'])
    df_merged['vix_return'] = df_merged['vix'].pct_change(1).fillna(0.0)
    return df_merged.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })

def write_forecast_to_cache(r_client, current_price, current_vix, current_realized_vol, forecasted_vol, expected_change, action, source_datetime):
    """Writes the forecast details to Upstash Redis cache with a 1-hour expiration."""
    result = {
        "ticker": "^NSEI",
        "current_price": float(current_price),
        "current_vix": float(current_vix),
        "current_realized_vol": float(current_realized_vol),
        "forecasted_vol_5h": float(forecasted_vol),
        "expected_change_pct": float(expected_change),
        "action": action,
        "date": source_datetime.strftime("%Y-%m-%d %H:%M:%S")
    }
    r_client.setex("nifty_forecast", 3600, json.dumps(result))
    return result

def get_next_market_candle(dt, steps=5):
    """Calculates the Nth subsequent Nifty trading candle datetime using pandas_market_calendars for NSE."""
    try:
        # Get NSE calendar
        nse = mcal.get_calendar('NSE')
        # We look ahead up to 20 calendar days to find the next trading days
        start_date = dt.date()
        end_date = start_date + datetime.timedelta(days=20)
        
        schedule = nse.schedule(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        # Valid trading days in index as pandas Timestamps (make naive)
        trading_days = schedule.index.tz_localize(None)
        
        # Hourly market times in IST
        market_times = [
            datetime.time(9, 15),
            datetime.time(10, 15),
            datetime.time(11, 15),
            datetime.time(12, 15),
            datetime.time(13, 15),
            datetime.time(14, 15),
            datetime.time(15, 15)
        ]
        
        # Build list of all valid candle datetimes starting from start_date
        valid_candles = []
        for day in trading_days:
            for t in market_times:
                valid_candles.append(datetime.datetime.combine(day.date(), t))
                
        # Find index of the first candle that is >= dt
        dt_naive = dt.replace(tzinfo=None)
        
        matching_idx = None
        for i, val in enumerate(valid_candles):
            if val == dt_naive:
                matching_idx = i
                break
        
        if matching_idx is not None:
            target_idx = matching_idx + steps
            if target_idx < len(valid_candles):
                return valid_candles[target_idx]
                
    except Exception as e:
        print(f"Error resolving calendar via pandas_market_calendars: {e}. Falling back to simple heuristic.")
        
    # Heuristic Fallback (skips weekends and wraps hourly candles)
    market_times = [
        datetime.time(9, 15),
        datetime.time(10, 15),
        datetime.time(11, 15),
        datetime.time(12, 15),
        datetime.time(13, 15),
        datetime.time(14, 15),
        datetime.time(15, 15)
    ]
    
    current_dt = dt.replace(tzinfo=None)
    for _ in range(steps):
        current_time = current_dt.time()
        try:
            idx = market_times.index(current_time)
        except ValueError:
            idx = -1
            for i, t in enumerate(market_times):
                if current_time <= t:
                    idx = i
                    break
            if idx == -1:
                current_dt = current_dt + datetime.timedelta(days=1)
                current_dt = current_dt.replace(hour=9, minute=15, second=0, microsecond=0)
                continue
        
        if idx < 6:
            next_time = market_times[idx + 1]
            current_dt = current_dt.replace(hour=next_time.hour, minute=next_time.minute, second=0, microsecond=0)
        else:
            current_dt = current_dt + datetime.timedelta(days=1)
            current_dt = current_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            
        while current_dt.weekday() >= 5:
            current_dt = current_dt + datetime.timedelta(days=1)
            
    return current_dt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS_VOL = [
    "rsi", "macd_diff_pct", "bb_width", "atr_pct", "hl_spread", "volume_delta", 
    "lagged_return_1", "vix", "vix_return", "realized_vol_5", "realized_vol_10", 
    "realized_vol_20", "sin_hour", "cos_hour", "sin_day", "cos_day"
]

class TemporalPriorAttention4h(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int = 42, bias_len: int = 6, bias_weight: float = 2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, 1))
        bias = torch.zeros(seq_len)
        bias[-bias_len:] = bias_weight
        self.bias = nn.Parameter(bias.unsqueeze(0), requires_grad=False)

    def forward(self, gru_out):
        raw_scores = torch.matmul(gru_out, self.weight)
        scores = raw_scores.squeeze(-1) + self.bias
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(gru_out * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionGRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalPriorAttention4h(hidden_dim, seq_len=42, bias_len=6, bias_weight=2.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, hn = self.gru(x)
        context, weights = self.attention(out)
        prediction = self.fc(context)
        return prediction.squeeze(-1)

# Helper function to generate features on the fly
def build_live_features(df_merged):
    df = df_merged.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff_pct'] = (macd - signal) / (close + 1e-9)
    
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
    
    df['realized_vol_5'] = df['lagged_return_1'].rolling(5).std()
    df['realized_vol_10'] = df['lagged_return_1'].rolling(10).std()
    df['realized_vol_20'] = df['lagged_return_1'].rolling(20).std()
    
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

# =====================================================================
# DATABASE WRITE OPERATIONS
# =====================================================================
def upsert_raw_market_data(cur, df_merged):
    """Upserts raw price and VIX candles into the database using bulk execute_values."""
    data = []
    for timestamp, row in df_merged.iterrows():
        vol = int(row['volume']) if not pd.isna(row['volume']) else 0
        data.append((
            timestamp, 
            float(row['open']), 
            float(row['high']), 
            float(row['low']), 
            float(row['close']), 
            vol, 
            float(row['vix'])
        ))
    
    query = """
        INSERT INTO nifty_vix_raw (datetime, open, high, low, close, volume, vix)
        VALUES %s
        ON CONFLICT (datetime) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vix = EXCLUDED.vix;
    """
    execute_values(cur, query, data)

def upsert_training_data(cur, df_features):
    """Upserts fully generated 16-feature records into the database using bulk execute_values."""
    data = []
    for timestamp, row in df_features.iterrows():
        data.append((
            timestamp, 
            float(row['close']), 
            float(row['rsi']), 
            float(row['macd_diff_pct']), 
            float(row['bb_width']), 
            float(row['atr_pct']), 
            float(row['hl_spread']), 
            float(row['volume_delta']), 
            float(row['lagged_return_1']), 
            float(row['vix']), 
            float(row['vix_return']), 
            float(row['realized_vol_5']), 
            float(row['realized_vol_10']), 
            float(row['realized_vol_20']), 
            float(row['close_fracdiff']), 
            float(row['sin_hour']), 
            float(row['cos_hour']), 
            float(row['sin_day']), 
            float(row['cos_day'])
        ))
        
    query = """
        INSERT INTO nifty_training_data 
        (datetime, close, rsi, macd_diff_pct, bb_width, atr_pct, hl_spread, volume_delta, 
         lagged_return_1, vix, vix_return, realized_vol_5, realized_vol_10, realized_vol_20, 
         close_fracdiff, sin_hour, cos_hour, sin_day, cos_day)
        VALUES %s
        ON CONFLICT (datetime) DO UPDATE SET
            close = EXCLUDED.close,
            rsi = EXCLUDED.rsi,
            macd_diff_pct = EXCLUDED.macd_diff_pct,
            bb_width = EXCLUDED.bb_width,
            atr_pct = EXCLUDED.atr_pct,
            hl_spread = EXCLUDED.hl_spread,
            volume_delta = EXCLUDED.volume_delta,
            lagged_return_1 = EXCLUDED.lagged_return_1,
            vix = EXCLUDED.vix,
            vix_return = EXCLUDED.vix_return,
            realized_vol_5 = EXCLUDED.realized_vol_5,
            realized_vol_10 = EXCLUDED.realized_vol_10,
            realized_vol_20 = EXCLUDED.realized_vol_20,
            close_fracdiff = EXCLUDED.close_fracdiff,
            sin_hour = EXCLUDED.sin_hour,
            cos_hour = EXCLUDED.cos_hour,
            sin_day = EXCLUDED.sin_day,
            cos_day = EXCLUDED.cos_day;
    """
    execute_values(cur, query, data)

# =====================================================================
# LIVE INFERENCE EXECUTION
# =====================================================================
def generate_live_inference():
    model_path = "models/attention_regressor.pt"
    scaler_path = "models/scaler_regressor.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model files missing. Train the model first.")
        
    model = AttentionGRURegressor(input_dim=16, hidden_dim=256, num_layers=3, dropout=0.19345).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(scaler_path, "rb") as f_in:
        scaler = pickle.load(f_in)
        
    # Query database for last ingested datetime and last 150 raw rows to optimize Lookback
    last_dt = None
    db_rows = []
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT MAX(datetime) FROM nifty_vix_raw;")
        last_dt = cur.fetchone()[0]
        if last_dt:
            # Query last 150 raw candles
            cur.execute("""
                SELECT datetime, open, high, low, close, volume, vix 
                FROM nifty_vix_raw 
                ORDER BY datetime DESC 
                LIMIT 150;
            """)
            db_rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as db_err:
        print(f"Could not retrieve last ingested candle from database: {db_err}")
        
    if last_dt and db_rows:
        # DB Warm-up: Convert rows to DataFrame
        df_db = pd.DataFrame(db_rows, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'vix'])
        df_db.set_index('datetime', inplace=True)
        df_db.index = pd.to_datetime(df_db.index)
        
        # Download only the last 3 days from Yahoo Finance to minimize latency
        start_str = (last_dt - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        print(f"Fetching live market data from Yahoo Finance since {start_str}...")
        df_nifty_new = yf.download("^NSEI", start=start_str, interval="1h", timeout=15)
        df_vix_new = yf.download("^INDIAVIX", start=start_str, interval="1h", timeout=15)
        
        df_merged_new = merge_nifty_vix(df_nifty_new, df_vix_new)
        
        # Combine database history and new Yahoo Finance data
        df_combined = pd.concat([df_db, df_merged_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
    else:
        # Full download fallback
        print("Fetching last 30 days of market data from Yahoo Finance...")
        df_nifty = yf.download("^NSEI", period="30d", interval="1h", timeout=15)
        df_vix = yf.download("^INDIAVIX", period="30d", interval="1h", timeout=15)
        df_combined = merge_nifty_vix(df_nifty, df_vix)
        
    # Build technical features
    df_features = build_live_features(df_combined)
    df_features = apply_fractional_differentiation(df_features)
    
    # Safety Check: Enforce minimum data length
    if len(df_features) < SEQ_LEN:
        raise ValueError(f"Not enough valid market rows for inference. Required: {SEQ_LEN}, got: {len(df_features)}")
    
    latest_X = df_features[FEATURE_COLS_VOL].values[-SEQ_LEN:]
    current_price = df_features['close'].values[-1]
    current_realized_vol = df_features['realized_vol_5'].values[-1] * 100.0
    current_vix = df_features['vix'].values[-1]
    
    # Scale inputs
    latest_X_scaled = scaler.transform(latest_X.reshape(-1, 16)).reshape(1, SEQ_LEN, 16)
    
    with torch.no_grad():
        latest_X_tensor = torch.FloatTensor(latest_X_scaled).to(device)
        forecasted_vol = model(latest_X_tensor).cpu().numpy()[0]
        
    # Guard against zero realized volatility denominator
    expected_change = ((forecasted_vol - current_realized_vol) / max(current_realized_vol, 1e-8)) * 100
    
    if forecasted_vol > (current_realized_vol * 1.50):
        action = "⚠️ CAUTION: Entering High Volatility. Reduce trade sizes / Buy puts."
    else:
        action = "✅ NORMAL: Market remains calm. Range-bound trading / standard sizing active."
        
    # Get exact timestamps for forecast mapping
    source_datetime = df_features.index[-1]
    target_datetime = get_next_market_candle(source_datetime, steps=5)
    
    # Filter only new data to write back to the database
    if last_dt:
        df_merged_to_write = df_combined[df_combined.index >= last_dt]
        df_features_to_write = df_features[df_features.index >= last_dt]
    else:
        df_merged_to_write = df_combined
        df_features_to_write = df_features
        
    # Single Transaction Database Ingestion
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 30000;")
            
        with conn:
            with conn.cursor() as cur:
                print("Updating raw database table...")
                upsert_raw_market_data(cur, df_merged_to_write)
                print("Updating training features database table...")
                upsert_training_data(cur, df_features_to_write)
                print("Logging forecast to database...")
                cur.execute("""
                    INSERT INTO volatility_forecasts 
                    (source_datetime, target_datetime, current_price, current_vix, current_realized_vol, forecasted_vol_5h, expected_change_pct, action, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'attention_gru_v1')
                    ON CONFLICT (ticker, source_datetime, model_version) DO UPDATE SET
                        target_datetime = EXCLUDED.target_datetime,
                        current_price = EXCLUDED.current_price,
                        current_vix = EXCLUDED.current_vix,
                        current_realized_vol = EXCLUDED.current_realized_vol,
                        forecasted_vol_5h = EXCLUDED.forecasted_vol_5h,
                        expected_change_pct = EXCLUDED.expected_change_pct,
                        action = EXCLUDED.action,
                        datetime = CURRENT_TIMESTAMP;
                """, (
                    source_datetime,
                    target_datetime,
                    float(current_price),
                    float(current_vix),
                    float(current_realized_vol),
                    float(forecasted_vol),
                    float(expected_change),
                    action
                ))
        print("Database transaction committed successfully.")
    except Exception as db_err:
        print(f"Database transaction failed, rolled back. Error: {db_err}")
        raise db_err
    finally:
        if conn:
            conn.close()
            
    # Active Redis Cache Invalidation / Refresh
    try:
        r_client = redis.Redis(**REDIS_CONFIG, socket_timeout=10, socket_connect_timeout=10)
        write_forecast_to_cache(
            r_client, current_price, current_vix, current_realized_vol, 
            forecasted_vol, expected_change, action, source_datetime
        )
        print("Redis cache updated successfully.")
    except Exception as cache_err:
        print(f"Failed to update Redis cache (forecast inserted to DB successfully): {cache_err}")
    
    print("\n" + "#"*70)
    print(f"  LIVE PRODUCTION SIGNAL FOR INDEX: ^NSEI (Nifty 50)")
    print(f"  Current Nifty 50 Price  : ₹{current_price:,.2f}")
    print(f"  Current India VIX Level : {current_vix:.2f}")
    print(f"  Current Realized Vol    : {current_realized_vol:.4f}%")
    print(f"  GRU Forecasted Vol (5h) : {forecasted_vol:.4f}%")
    print(f"  Expected Volatility Change: {expected_change:+.2f}%")
    print(f"  Action                  : {action}")
    print("#"*70)

if __name__ == "__main__":
    generate_live_inference()