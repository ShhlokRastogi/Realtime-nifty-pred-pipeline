import os
import time
import pickle
import psycopg2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from config import SEQ_LEN, FFD_D, FFD_MAX_LAGS, DB_CONFIG

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
def upsert_raw_market_data(conn, df_merged):
    """Upserts raw price and VIX candles into the database."""
    cur = conn.cursor()
    for timestamp, row in df_merged.iterrows():
        # Handle volume conversion to standard python int/float
        vol = int(row['volume']) if not pd.isna(row['volume']) else 0
        cur.execute("""
            INSERT INTO nifty_vix_raw (datetime, open, high, low, close, volume, vix)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (datetime) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vix = EXCLUDED.vix;
        """, (
            timestamp, 
            float(row['open']), 
            float(row['high']), 
            float(row['low']), 
            float(row['close']), 
            vol, 
            float(row['vix'])
        ))
    conn.commit()
    cur.close()

def upsert_training_data(conn, df_features):
    """Upserts fully generated 16-feature records into the database."""
    cur = conn.cursor()
    for timestamp, row in df_features.iterrows():
        cur.execute("""
            INSERT INTO nifty_training_data 
            (datetime, close, rsi, macd_diff_pct, bb_width, atr_pct, hl_spread, volume_delta, 
             lagged_return_1, vix, vix_return, realized_vol_5, realized_vol_10, realized_vol_20, 
             close_fracdiff, sin_hour, cos_hour, sin_day, cos_day)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        """, (
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
    conn.commit()
    cur.close()

def log_forecast_to_db(conn, price, vix, realized_vol, forecasted_vol, change_pct, action):
    """Inserts prediction row into database."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO volatility_forecasts 
        (current_price, current_vix, current_realized_vol, forecasted_vol_5h, expected_change_pct, action)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        float(price), 
        float(vix), 
        float(realized_vol), 
        float(forecasted_vol), 
        float(change_pct), 
        action
    ))
    conn.commit()
    cur.close()

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
        
    print("Fetching live market data from Yahoo Finance...")
    df_nifty = yf.download("^NSEI", period="15d", interval="1h")
    df_vix = yf.download("^INDIAVIX", period="15d", interval="1h")
    
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    df_nifty.index = df_nifty.index.tz_localize(None)
    df_vix.index = df_vix.index.tz_localize(None)
    
    df_vix_close = df_vix[['Close']].rename(columns={'Close': 'vix'})
    df_merged = df_nifty.join(df_vix_close, how='inner')
    df_merged['vix'] = df_merged['vix'].ffill().bfill()
    df_merged['vix_return'] = df_merged['vix'].pct_change(1).fillna(0.0)
    df_merged = df_merged.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    # 1. Update Raw Market Table in Supabase
    conn = psycopg2.connect(**DB_CONFIG)
    print("Updating raw database table...")
    upsert_raw_market_data(conn, df_merged)
    
    # 2. Build Technical Features
    df_features = build_live_features(df_merged)
    df_features = apply_fractional_differentiation(df_features)
    
    # 3. Update Training Table in Supabase (with freshly generated features)
    print("Updating training features database table...")
    upsert_training_data(conn, df_features)
    
    latest_X = df_features[FEATURE_COLS_VOL].values[-SEQ_LEN:]
    current_price = df_features['close'].values[-1]
    current_realized_vol = df_features['realized_vol_5'].values[-1] * 100.0
    current_vix = df_features['vix'].values[-1]
    
    # Scale inputs
    latest_X_scaled = scaler.transform(latest_X.reshape(-1, 16)).reshape(1, SEQ_LEN, 16)
    
    with torch.no_grad():
        latest_X_tensor = torch.FloatTensor(latest_X_scaled).to(device)
        forecasted_vol = model(latest_X_tensor).cpu().numpy()[0]
        
    expected_change = ((forecasted_vol - current_realized_vol)/current_realized_vol)*100
    
    if forecasted_vol > (current_realized_vol * 1.50):
        action = "⚠️ CAUTION: Entering High Volatility. Reduce trade sizes / Buy puts."
    else:
        action = "✅ NORMAL: Market remains calm. Range-bound trading / standard sizing active."
        
    # 4. Log the forecast to Supabase
    log_forecast_to_db(conn, current_price, current_vix, current_realized_vol, forecasted_vol, expected_change, action)
    conn.close()
    
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