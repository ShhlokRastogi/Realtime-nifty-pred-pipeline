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


import joblib
import redis
import os
import json
from config import REDIS_CONFIG
from prometheus_client import Gauge, Counter, REGISTRY

# Register/load Prometheus gauges inside the shared process memory
if 'crypto_mlops_drift_live_accuracy' in REGISTRY._names_to_collectors:
    ACCURACY_GAUGE = REGISTRY._names_to_collectors['crypto_mlops_drift_live_accuracy']
else:
    ACCURACY_GAUGE = Gauge("live_accuracy", "Rolling accuracy of the model", ["ticker"], namespace="crypto_mlops", subsystem="drift")

if 'crypto_mlops_drift_drift_detected' in REGISTRY._names_to_collectors:
    DRIFT_GAUGE = REGISTRY._names_to_collectors['crypto_mlops_drift_drift_detected']
else:
    DRIFT_GAUGE = Gauge("drift_detected", "Performance drift detected (accuracy < 50%)", ["ticker"], namespace="crypto_mlops", subsystem="drift")

def evaluate_live_accuracy(ticker: str, current_time: pd.Timestamp, current_close: float):
    """
    Checks if a prediction was logged for the previous candle. 
    If yes, evaluates if it was correct, logs to Redis history, and updates gauges.
    """
    r = redis.Redis(**REDIS_CONFIG)
    
    # 1. We look back 15 minutes to find the previous candle timestamp
    prev_time = current_time - pd.Timedelta(minutes=15)
    prev_time_str = str(prev_time)
    
    pred_key = f"pred:{ticker}:{prev_time_str}"
    close_key = f"close:{ticker}:{prev_time_str}"
    
    saved_pred = r.get(pred_key)
    saved_close = r.get(close_key)
    
    if saved_pred and saved_close:
        prediction = saved_pred.decode()
        prev_close = float(saved_close.decode())
        
        # Calculate true label
        actual = "UP" if current_close > prev_close else "DOWN"
        is_correct = 1 if prediction == actual else 0
        
        print(f"  [EVAL] {ticker}: Pred at {prev_time_str} was {prediction}, Actual was {actual} ({'CORRECT' if is_correct else 'INCORRECT'})")
        
        # Push outcome to Redis rolling window (max 100)
        history_key = f"history:{ticker}"
        r.lpush(history_key, is_correct)
        r.ltrim(history_key, 0, 99) # Keep last 100 outcomes
        
        # Clean up Redis prediction keys
        r.delete(pred_key)
        r.delete(close_key)
        
    # 2. Retrieve history and calculate rolling accuracy
    history_key = f"history:{ticker}"
    outcomes = r.lrange(history_key, 0, -1)
    
    if outcomes:
        outcomes_list = [int(x.decode()) for x in outcomes]
        rolling_acc = sum(outcomes_list) / len(outcomes_list)
        ACCURACY_GAUGE.labels(ticker=ticker).set(rolling_acc)
        print(f"  [ACCURACY] {ticker} Rolling Accuracy (last {len(outcomes_list)} runs): {rolling_acc:.4f}")
        
        # Performance Drift Trigger: If we have at least 10 outcomes and accuracy falls below 50%
        if len(outcomes_list) >= 10 and rolling_acc < 0.50:
            print(f"  [DRIFT] Alert! {ticker} accuracy has dropped below 50%. Setting drift_detected=1")
            DRIFT_GAUGE.labels(ticker=ticker).set(1.0)
        else:
            DRIFT_GAUGE.labels(ticker=ticker).set(0.0)
    else:
        # Default accuracy and drift to neutral state if no history exists yet
        ACCURACY_GAUGE.labels(ticker=ticker).set(0.53) # Incumbent baseline
        DRIFT_GAUGE.labels(ticker=ticker).set(0.0)

def generate_live_prediction(ticker: str, current_time: pd.Timestamp, current_close: float):
    """
    Generates a prediction using the active model for the next candle interval, 
    and saves the state to Redis to be evaluated on the next run.
    """
    r = redis.Redis(**REDIS_CONFIG)
    
    # Load model and cache
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "active_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "..", "models", "xgb_model.pkl")
        
    if os.path.exists(model_path):
        try:
            # Retrieve latest computed feature set from Redis cache
            cache_raw = r.get(f"features:{ticker}")
            if cache_raw:
                payload = json.loads(cache_raw.decode())
                features = payload["features"]
                feature_cols = [
                    "rsi", "sma_crossover", "rolling_volatility",
                    "volume_delta", "lagged_return_1", "lagged_return_3", "lagged_return_5"
                ]
                
                # Make prediction
                df_feat = pd.DataFrame([features])[feature_cols]
                model = joblib.load(model_path)
                pred_val = model.predict(df_feat)[0]
                prediction_string = "UP" if pred_val == 1 else "DOWN"
                
                # Log prediction and close price for this timestamp to be evaluated on the next run
                current_time_str = str(current_time)
                r.set(f"pred:{ticker}:{current_time_str}", prediction_string)
                r.set(f"close:{ticker}:{current_time_str}", current_close)
                print(f"  [PRED] Logged {ticker} prediction for next candle: {prediction_string}")
        except Exception as e:
            print(f"  Error generating poller prediction for {ticker}: {e}")

def poll_once():
    """Fetches new prices, saves to DB, updates features, evaluates accuracy, and caches to Redis."""
    print("\n=== Polling cycle started ===")
    for ticker in TICKERS:
        try:
            print(f"Fetching latest data for {ticker} from Coinbase...")
            # Fetch latest 15-minute candles from Coinbase
            df = fetch_latest_candles_coinbase(ticker, limit=5)
            
            # Upsert into Postgres ohlcv table
            save_to_postgres(df, ticker)
            print(f"  Successfully saved/updated {len(df)} rows in database.")
            
            # Retrieve the latest closed candle details
            latest_row = df.iloc[-1]
            latest_time = latest_row.name
            latest_close = float(latest_row["Close"])
            
            # Step 1: Evaluate accuracy of the previous prediction
            evaluate_live_accuracy(ticker, latest_time, latest_close)
            
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")
            
    # Step 2: Trigger feature store to recompute and write features
    try:
        print("Triggering Feature Store run (incremental mode: limit=100)...")
        run_feature_store(limit=100)
        
        # Step 3: Generate prediction for the next candle and save state in Redis
        for ticker in TICKERS:
            # Query maximum date from Postgres to match what the feature store just cached
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT date, close FROM ohlcv WHERE ticker = %s ORDER BY date DESC LIMIT 1", (ticker,))
            latest = cur.fetchone()
            cur.close()
            conn.close()
            if latest:
                generate_live_prediction(ticker, latest[0], float(latest[1]))
                
    except Exception as e:
        print(f"  Error during Feature Store run: {e}")
    print("=== Polling cycle complete ===")


def main():
    print("Starting Live Polling Service (Interval: 60s)...")
    import time
    while True:
        poll_once()
        time.sleep(60)


if __name__ == "__main__":
    main()
