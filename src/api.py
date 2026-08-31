import os
import sys
import json
import redis
import psycopg2
import threading
import time
import datetime
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from config import REDIS_CONFIG, DB_CONFIG, TICKERS
from prometheus_client import REGISTRY, Gauge, generate_latest, CONTENT_TYPE_LATEST
from poll import generate_live_inference
from drift import monitor_accuracy_drift

# Redirect stdout and stderr to a log file so we can view background thread tracebacks via /logs
class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

if os.getenv("TESTING") != "true":
    try:
        Tee("poller.log")
    except Exception as log_err:
        print(f"Failed to initialize log file redirection: {log_err}")

# App initialization
app = FastAPI(title="Nifty 50 Volatility Forecast API")

# Connect to Redis Upstash for caching
r = redis.Redis(**REDIS_CONFIG)

# =====================================================================
# BACKGROUND POLLER DAEMON THREAD
# =====================================================================
def acquire_postgres_lock(conn):
    """Acquires a database-backed poller lock using lock_store table, with a 50-minute timeout."""
    try:
        with conn.cursor() as cur:
            # Delete expired locks
            cur.execute("""
                DELETE FROM lock_store 
                WHERE lock_key = 'poller_lock' 
                  AND locked_at < CURRENT_TIMESTAMP - INTERVAL '50 minutes';
            """)
            # Try to insert
            cur.execute("""
                INSERT INTO lock_store (lock_key, locked_at) 
                VALUES ('poller_lock', CURRENT_TIMESTAMP) 
                ON CONFLICT (lock_key) DO NOTHING;
            """)
            success = cur.rowcount > 0
            return success
    except Exception as lock_err:
        print(f"Postgres Lock acquisition query failed: {lock_err}")
        return False

def release_postgres_lock(conn):
    """Releases the database-backed poller lock."""
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM lock_store WHERE lock_key = 'poller_lock';")
    except Exception as lock_err:
        print(f"Postgres Lock release query failed: {lock_err}")

def poller_loop():
    print("=== Background Poller & Drift Monitor Thread Initialized ===")
    while True:
        # 1. Check last forecast timestamp from database
        last_forecast_time = None
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 10000;")
                cur.execute("SELECT MAX(source_datetime) FROM volatility_forecasts;")
                last_forecast_time = cur.fetchone()[0]
        except Exception as db_err:
            print(f"Could not check last forecast timestamp: {db_err}")
        finally:
            if conn:
                conn.close()
                
        if last_forecast_time:
            # If last forecast was within 50 minutes, skip to avoid duplicates
            time_elapsed = datetime.datetime.now() - last_forecast_time
            if time_elapsed < datetime.timedelta(minutes=50):
                print(f"Last forecast was generated {time_elapsed.seconds // 60} minutes ago. Skipping this poller cycle.")
                time.sleep(300)  # Sleep 5 minutes and check again
                continue
                
        # 2. Acquire Postgres lock to prevent race conditions during scaling
        # (This is 100% database-backed, concurrency-safe, and fails closed)
        lock_acquired = False
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 10000;")
            lock_acquired = acquire_postgres_lock(conn)
            if lock_acquired:
                conn.commit()
        except Exception as conn_err:
            print(f"Could not connect to database for lock check: {conn_err}. Failing closed to prevent duplicate runs.")
            lock_acquired = False
        finally:
            if conn:
                conn.close()
                
        if not lock_acquired:
            print("Another poller worker holds the Postgres lock or database is down. Skipping this poller cycle.")
            time.sleep(300)  # Sleep 5 minutes and check again
            continue

        # 3. Update gauges and execute live inference & drift monitor
        try:
            # Update Prometheus Gauges from Supabase first so the dashboard displays real history immediately
            try:
                update_prometheus_metrics()
            except Exception as e:
                print(f"Error updating Prometheus metrics on startup: {e}")

            print("Executing hourly live data ingestion, feature generation, and prediction...")
            generate_live_inference()
            
            print("Executing hourly performance metrics and drift monitoring check...")
            monitor_accuracy_drift(window_hours=100)
            
            # Update again after a successful run
            update_prometheus_metrics()
            
        except Exception as e:
            print(f"Error in background execution thread: {e}")
        finally:
            # Release Postgres lock so the next poller run can proceed
            conn = None
            try:
                conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = 10000;")
                release_postgres_lock(conn)
                conn.commit()
            except Exception as release_err:
                print(f"Failed to release Postgres lock: {release_err}")
            finally:
                if conn:
                    conn.close()
                    
        time.sleep(3600)  # Run hourly

def update_prometheus_metrics():
    """Reads latest forecasts and drift metrics from Supabase and updates Prometheus."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 30000;")
        
        # 1. Fetch latest forecast (ordered by source_datetime DESC)
        cur.execute("""
            SELECT current_price, current_vix, current_realized_vol, forecasted_vol_5h, expected_change_pct
            FROM volatility_forecasts
            ORDER BY source_datetime DESC
            LIMIT 1;
        """)
        f_row = cur.fetchone()
        
        # 2. Fetch latest drift metrics
        cur.execute("""
            SELECT directional_accuracy, mean_absolute_error, r2_score, drift_detected
            FROM model_drift_metrics
            ORDER BY calculated_at DESC
            LIMIT 1;
        """)
        d_row = cur.fetchone()
        
        cur.close()
        
        if f_row:
            NIFTY_PRICE.set(float(f_row[0]))
            INDIA_VIX.set(float(f_row[1]))
            REALIZED_VOLATILITY.set(float(f_row[2]))
            FORECASTED_VOLATILITY.set(float(f_row[3]))
            EXPECTED_VOL_CHANGE.set(float(f_row[4]))
            
        if d_row:
            DIRECTIONAL_ACCURACY.set(float(d_row[0]))
            MAE_LOSS.set(float(d_row[1]))
            R2_SCORE.set(float(d_row[2]))
            DRIFT_GAUGE.set(1.0 if d_row[3] else 0.0)
            
        print("Prometheus gauges updated.")
    except Exception as e:
        print(f"Error updating Prometheus metrics: {e}")
    finally:
        if conn:
            conn.close()

@app.on_event("startup")
def startup_event():
    # If in testing mode, skip database initialization and poller thread
    if os.getenv("TESTING") == "true":
        print("=== Testing Environment Detected: Skipping Background Poller ===")
        return

    # Automatically create missing tables in Supabase
    try:
        from db_schema import initialize_database_schema
        initialize_database_schema()
    except Exception as e:
        print(f"Error initializing database schema on startup: {e}")

    # Run the background daemon poller
    t = threading.Thread(target=poller_loop, daemon=True)
    t.start()
    print("=== Background Poller Thread Started ===")

# =====================================================================
# PROMETHEUS METRICS CONFIGURATION
# =====================================================================
# Ensure we don't crash on Uvicorn live-reload by registering metric collectors safely
NIFTY_PRICE = REGISTRY._names_to_collectors.get('nifty_price') or Gauge("nifty_price", "Current Nifty 50 index price")
INDIA_VIX = REGISTRY._names_to_collectors.get('india_vix') or Gauge("india_vix", "Current India VIX fear index level")
REALIZED_VOLATILITY = REGISTRY._names_to_collectors.get('realized_volatility') or Gauge("realized_volatility", "Nifty realized volatility percentage")
FORECASTED_VOLATILITY = REGISTRY._names_to_collectors.get('forecasted_volatility') or Gauge("forecasted_volatility", "Attention GRU forecasted volatility percentage")
EXPECTED_VOL_CHANGE = REGISTRY._names_to_collectors.get('expected_volatility_change_pct') or Gauge("expected_volatility_change_pct", "Predicted change in volatility percentage")
DIRECTIONAL_ACCURACY = REGISTRY._names_to_collectors.get('directional_accuracy') or Gauge("directional_accuracy", "Current rolling directional accuracy of the model")
MAE_LOSS = REGISTRY._names_to_collectors.get('mae_loss') or Gauge("mae_loss", "Current mean absolute error of the model")
R2_SCORE = REGISTRY._names_to_collectors.get('r2_score') or Gauge("r2_score", "Current R-squared variance explained score")
DRIFT_GAUGE = REGISTRY._names_to_collectors.get('drift_detected') or Gauge("drift_detected", "Flag indicating model drift (1=Yes, 0=No)")

# =====================================================================
# API ENDPOINTS
# =====================================================================

def fetch_latest_nifty_forecast():
    """Helper to fetch the latest Nifty 50 volatility forecast from cache or database."""
    # If in testing environment, return mock Nifty volatility schema immediately
    if os.getenv("TESTING") == "true":
        return {
            "ticker": "^NSEI",
            "current_price": 24252.00,
            "current_vix": 11.22,
            "current_realized_vol": 0.0785,
            "forecasted_vol_5h": 0.1638,
            "expected_change_pct": 108.64,
            "action": "⚠️ CAUTION: Entering High Volatility.",
            "date": "2026-08-24 00:00:00"
        }

    # Check cache first (safely fallback to DB if Redis is offline)
    try:
        cached_data = r.get("nifty_forecast")
        if cached_data:
            print("Returning cached volatility forecast from Upstash Redis...")
            return json.loads(cached_data)
    except Exception as cache_err:
        print(f"Redis Cache connection failed (falling back to database): {cache_err}")
        
    # Query Supabase for the latest forecast (ordered by source_datetime DESC)
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 30000;")
        cur.execute("""
            SELECT current_price, current_vix, current_realized_vol, forecasted_vol_5h, expected_change_pct, action, source_datetime
            FROM volatility_forecasts
            ORDER BY source_datetime DESC
            LIMIT 1;
        """)
        row = cur.fetchone()
        cur.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="No predictions found in database.")
            
        result = {
            "ticker": "^NSEI",
            "current_price": float(row[0]),
            "current_vix": float(row[1]),
            "current_realized_vol": float(row[2]),
            "forecasted_vol_5h": float(row[3]),
            "expected_change_pct": float(row[4]),
            "action": row[5],
            "date": row[6].strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Cache in Redis safely (don't crash if Redis write fails)
        try:
            r.setex("nifty_forecast", 3600, json.dumps(result))
        except Exception as cache_err:
            print(f"Failed to save forecast to Redis cache: {cache_err}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/predict/nifty")
def predict_nifty():
    """Dedicated endpoint to get the live Nifty 50 volatility forecast."""
    return fetch_latest_nifty_forecast()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/logs")
def get_logs(api_key: str = None):
    """Exposes the last 200 lines of standard output/error logs for MLOps diagnostics."""
    if os.getenv("TESTING") != "true":
        expected_key = os.getenv("ADMIN_API_KEY")
        if not expected_key:
            expected_key = "default_secure_admin_key"
        if api_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized log access.")
        
    if not os.path.exists("poller.log"):
        return Response("No logs recorded yet.", media_type="text/plain")
    try:
        with open("poller.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
        return Response("".join(lines[-200:]), media_type="text/plain")
    except Exception as e:
        return Response(f"Failed to read logs: {e}", media_type="text/plain")

@app.get("/metrics")
def metrics():
    """Returns raw Prometheus scraping payload."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/promote", response_class=HTMLResponse)
def promote_page():
    """Renders GitOps model comparison page showing active model details and version history."""
    active_model = None
    history = []
    
    # 1. Fetch live metrics from local metrics JSON or database
    try:
        # Load local metrics backup
        if os.path.exists("models/metrics.json"):
            with open("models/metrics.json", "r") as f:
                meta = json.load(f)
                active_model = {
                    "version": "1.0.0",
                    "model_type": "Attention-GRU Regressor",
                    "r2": meta.get("r2", 26.47),
                    "mae": meta.get("mae", 0.1213),
                    "accuracy": meta.get("dir_accuracy", 75.81),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
    except Exception:
        pass
        
    # Load history from the Supabase model_drift_metrics table
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, calculated_at, directional_accuracy, mean_absolute_error, r2_score, drift_detected
            FROM model_drift_metrics
            ORDER BY calculated_at DESC
            LIMIT 5;
        """)
        rows = cur.fetchall()
        for r in rows:
            history.append({
                "id": r[0],
                "timestamp": r[1].strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": float(r[2]),
                "mae": float(r[3]),
                "r2": float(r[4]),
                "drift": "DRIFT WARNING" if r[5] else "HEALTHY"
            })
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database error loading drift history: {e}")
        
    # Default fallbacks if database table is fresh/empty
    if not active_model:
        active_model = {
            "version": "1.0.0",
            "model_type": "Attention-GRU Regressor",
            "r2": 26.47,
            "mae": 0.1213,
            "accuracy": 75.81,
            "timestamp": "N/A"
        }
        
    if not history:
        history = [{
            "id": 1,
            "timestamp": "N/A",
            "accuracy": 75.81,
            "mae": 0.1213,
            "r2": 26.47,
            "drift": "HEALTHY"
        }]

    history_html = ""
    for h in history:
        status_class = h["drift"].lower().replace(" ", "_")
        history_html += f"""
        <tr>
            <td>Eval Run #{h["id"]}</td>
            <td style="color: #10b981; font-weight: 600;">{h["accuracy"]:.2f}%</td>
            <td>{h["mae"]:.4f}%</td>
            <td>{h["r2"]:.2f}%</td>
            <td style="font-size: 0.85rem; color: #94a3b8;">{h["timestamp"]}</td>
            <td><span class="status-badge {status_class}">{h["drift"]}</span></td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nifty Volatility Pipeline Promotion Gate</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #0f172a;
                color: #f8fafc;
                margin: 0;
                padding: 40px;
            }}
            .container {{
                max-width: 950px;
                margin: 0 auto;
            }}
            h1 {{
                font-size: 2.2rem;
                margin-bottom: 5px;
                font-weight: 700;
            }}
            h2 {{
                font-size: 1.5rem;
                margin-top: 40px;
                margin-bottom: 15px;
                font-weight: 600;
                border-bottom: 1px solid #334155;
                padding-bottom: 10px;
                color: #e2e8f0;
            }}
            .subtitle {{
                color: #94a3b8;
                margin-bottom: 30px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background-color: #1e293b;
                border-radius: 12px;
                padding: 24px;
                border: 1px solid #334155;
            }}
            .card.staging {{
                border-color: #3b82f6;
                box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
            }}
            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-bottom: 15px;
            }}
            .badge.prod {{
                background-color: #10b98120;
                color: #10b981;
            }}
            .badge.stage {{
                background-color: #3b82f620;
                color: #3b82f6;
            }}
            .version {{
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 15px;
            }}
            .metric-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #334155;
            }}
            .metric-label {{
                color: #94a3b8;
            }}
            .metric-value {{
                font-weight: 600;
            }}
            .button {{
                display: block;
                width: 100%;
                padding: 15px;
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                text-align: center;
                text-decoration: none;
                transition: background-color 0.2s;
            }}
            .button:hover {{
                background-color: #2563eb;
            }}
            .no-model {{
                color: #64748b;
                text-align: center;
                padding: 40px 0;
                font-size: 0.95rem;
                line-height: 1.5;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background-color: #1e293b;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid #334155;
                margin-bottom: 40px;
            }}
            th, td {{
                padding: 14px 18px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }}
            th {{
                background-color: #1e293b;
                color: #94a3b8;
                font-weight: 600;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            .status-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.03em;
            }}
            .status-badge.healthy {{
                background-color: #10b98120;
                color: #10b981;
            }}
            .status-badge.drift_warning {{
                background-color: #ef444420;
                color: #ef4444;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Promotion & Monitoring Gate</h1>
            <p class="subtitle">Evaluate Nifty 50 Volatility model performance and promote candidate runs to production using Git-Ops.</p>
            
            <div class="grid">
                <!-- Incumbent Card -->
                <div class="card">
                    <span class="badge prod">PRODUCTION (Active Model)</span>
                    <div class="version">Version {active_model["version"]}</div>
                    <div class="metric-row">
                        <span class="metric-label">Model Type</span>
                        <span class="metric-value">{active_model["model_type"]}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">R2 Score (Variance Explained)</span>
                        <span class="metric-value" style="color: #3b82f6;">{active_model["r2"]:.2f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">MAE Loss</span>
                        <span class="metric-value">{active_model["mae"]:.4f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Directional Accuracy</span>
                        <span class="metric-value" style="color: #10b981;">{active_model["accuracy"]:.2f}%</span>
                    </div>
                    <div class="metric-row" style="border-bottom: none;">
                        <span class="metric-label">Registered At</span>
                        <span class="metric-value" style="font-size: 0.8rem;">{active_model["timestamp"]}</span>
                    </div>
                </div>
                
                <!-- Candidate Card -->
                <div class="card staging">
                    <span class="badge stage">STAGING (Git-Ops Evaluation)</span>
                    <div class="version">Candidate Staging PR</div>
                    <div class="no-model">
                        Model candidates are submitted as GitHub Pull Requests.<br><br>
                        Merging a PR automatically runs PyTest checks, DVC pull, registers the weights to MLflow, and deploys it live!
                    </div>
                </div>
            </div>
            
            <a href="https://github.com/ShhlokRastogi/self-healing-crypto-pipeline/pulls" target="_blank" class="button">
                Review and Merge Pull Requests on GitHub
            </a>
            
            <h2>Model Drift & Performance History (Supabase)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Evaluation Run</th>
                        <th>Directional Accuracy</th>
                        <th>MAE Loss</th>
                        <th>R2 Score</th>
                        <th>Evaluated At</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {history_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html_content

    return html_content

