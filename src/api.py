import json
import joblib
import redis
from fastapi import FastAPI, HTTPException, Response
from config import REDIS_CONFIG, TICKERS
from prometheus_client import REGISTRY, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from drift import run_drift_check, FEATURE_COLS

app = FastAPI(title="Crypto Price Prediction API")
import os
import mlflow
import joblib

# Load the local backup model dynamically relative to this file's location
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "active_model.pkl")
if not os.path.exists(model_path):
    model_path = os.path.join(base_dir, "..", "models", "xgb_model.pkl")
model = joblib.load(model_path)
r = redis.Redis(**REDIS_CONFIG)

import threading
import time
from poll import poll_once

def poller_loop():
    print("=== Background Poller Thread Initialized ===")
    while True:
        try:
            poll_once()
        except Exception as e:
            print(f"Error in background poller execution: {e}")
        time.sleep(60)

@app.on_event("startup")
def startup_event():
    # Run the poller loop in a background daemon thread
    t = threading.Thread(target=poller_loop, daemon=True)
    t.start()
    print("=== Background Poller Thread Started ===")


# Check if metrics are already registered (to prevent Uvicorn reload crashes)
if 'crypto_mlops_drift_drift_detected' in REGISTRY._names_to_collectors:
    DRIFT_GAUGE = REGISTRY._names_to_collectors['crypto_mlops_drift_drift_detected']
else:
    DRIFT_GAUGE = Gauge(
        "drift_detected",
        "Drift detected for a given ticker",
        ["ticker"],
        namespace="crypto_mlops",
        subsystem="drift",
    )
    # TO QUERY IN PROMQL USE FORMAT: crypto_mlops_drift_drift_detected{ticker="BTC-USD"}

if 'crypto_mlops_drift_p_value' in REGISTRY._names_to_collectors:
    PVALUE_GAUGE = REGISTRY._names_to_collectors['crypto_mlops_drift_p_value']
else:
    PVALUE_GAUGE = Gauge(
        "p_value",
        "P-value for a given feature and ticker",
        ["ticker", "feature"],
        namespace="crypto_mlops",
        subsystem="drift",
    )
    # TO QUERY IN PROMQL USE FORMAT: crypto_mlops_drift_p_value{ticker="BTC-USD",feature="lagged_return_1"}

if 'crypto_mlops_prediction_predictions_total' in REGISTRY._names_to_collectors:
    PREDICTION_COUNTER = REGISTRY._names_to_collectors['crypto_mlops_prediction_predictions_total']
else:
    PREDICTION_COUNTER = Counter(
        "predictions_total",
        "Total number of predictions made",
        ["ticker", "prediction"],
        namespace="crypto_mlops",
        subsystem="prediction",
    )
    # TO QUERY IN PROMQL USE FORMAT: crypto_mlops_prediction_predictions_total{ticker="BTC-USD",prediction="UP"}

feature_cols = [
    "lagged_return_1", "lagged_return_3", "lagged_return_5",
    "sma_crossover", "rolling_volatility", "volume_delta", "rsi"
]

@app.get("/predict/{ticker}")
def predict(ticker: str):
    raw = r.get(f"features:{ticker}")
    if raw is None:
        raise HTTPException(status_code=404, detail=f"No features found for {ticker}")
    
    data = json.loads(raw)
    features = data["features"]
    values = [[features[col] for col in feature_cols]]
    
    prediction = model.predict(values)[0]
    probabilities = model.predict_proba(values)[0]
    confidence = float(max(probabilities))
    
    # ── NEW: Increment prediction count ──
    prediction_label = "UP" if prediction == 1 else "DOWN"
    PREDICTION_COUNTER.labels(ticker=ticker, prediction=prediction_label).inc()
    
    return {
        "ticker": ticker,
        "prediction": prediction_label,
        "confidence": confidence,
        "date": data["date"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    import gc
    # Loop through our active tickers (BTC-USD, ETH-USD)
    for t in TICKERS:
        # Run the drift calculations
        drift_result = run_drift_check(t)
        
        if drift_result.get("status") == "success":
            # Set overall drift status (1.0 if True, 0.0 if False)
            drift_detected = 1.0 if drift_result["drift_detected"] else 0.0
            DRIFT_GAUGE.labels(ticker=t).set(drift_detected)
            
            # Set the individual p-values for each feature
            p_values = drift_result["p_values"]
            for feature_name, p_val in p_values.items():
                PVALUE_GAUGE.labels(ticker=t, feature=feature_name).set(p_val)
                
        # Clean up memory immediately after each ticker calculation
        del drift_result
        gc.collect()
                
    # Return the metrics in the raw text format that Prometheus understands
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


from fastapi.responses import HTMLResponse

@app.get("/promote", response_class=HTMLResponse)
def promote_page():
    """Renders a beautiful model comparison page showing active model details and version history from Postgres."""
    import psycopg2
    from config import DB_CONFIG
    import json
    
    # 1. Query the database for the active model and history
    active_model = None
    history = []
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Fetch the active model (marked 'active')
        cur.execute("""
            SELECT version, model_type, accuracy, incumbent_accuracy, dummy_accuracy, parameters, timestamp
            FROM models
            WHERE status = 'active'
            ORDER BY timestamp DESC
            LIMIT 1;
        """)
        active_row = cur.fetchone()
        if active_row:
            active_model = {
                "version": active_row[0],
                "model_type": active_row[1],
                "accuracy": active_row[2],
                "incumbent_accuracy": active_row[3],
                "dummy_accuracy": active_row[4],
                "parameters": active_row[5] if isinstance(active_row[5], dict) else json.loads(active_row[5] or "{}"),
                "timestamp": active_row[6].strftime("%Y-%m-%d %H:%M:%S") if active_row[6] else "N/A"
            }
            
        # Fetch the last 5 registered models (history)
        cur.execute("""
            SELECT version, model_type, accuracy, incumbent_accuracy, dummy_accuracy, timestamp, status
            FROM models
            ORDER BY timestamp DESC
            LIMIT 5;
        """)
        history_rows = cur.fetchall()
        for h in history_rows:
            history.append({
                "version": h[0],
                "model_type": h[1],
                "accuracy": h[2],
                "incumbent_accuracy": h[3],
                "dummy_accuracy": h[4],
                "timestamp": h[5].strftime("%Y-%m-%d %H:%M:%S") if h[5] else "N/A",
                "status": h[6]
            })
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database error loading model history: {e}")
        
    # 2. Fall back to local metadata JSON if database is empty
    if not active_model:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        meta_path = os.path.join(base_dir, "..", "models", "model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    active_model = {
                        "version": meta.get("version", 1),
                        "model_type": meta.get("model_type", "xgboost"),
                        "accuracy": meta.get("accuracy", 0.5248),
                        "incumbent_accuracy": meta.get("incumbent_accuracy", 0.0),
                        "dummy_accuracy": meta.get("dummy_accuracy", 0.4898),
                        "timestamp": meta.get("timestamp", "N/A"),
                        "parameters": meta.get("parameters", {})
                    }
            except Exception:
                pass
                
    # Define fallback defaults if metadata read also failed
    if not active_model:
        active_model = {
            "version": 1,
            "model_type": "xgboost",
            "accuracy": 0.5248,
            "incumbent_accuracy": 0.0,
            "dummy_accuracy": 0.4898,
            "timestamp": "2026-08-15 12:00:00",
            "parameters": {"max_depth": 3, "n_estimators": 69}
        }
        history = [{
            "version": 1,
            "model_type": "xgboost",
            "accuracy": 0.5248,
            "incumbent_accuracy": 0.0,
            "dummy_accuracy": 0.4898,
            "timestamp": "2026-08-15 12:00:00",
            "status": "active"
        }]

    # Render History Rows HTML
    history_html = ""
    for h in history:
        status_class = h["status"].lower()
        history_html += f"""
        <tr>
            <td>Version {h["version"]}</td>
            <td style="text-transform: uppercase; font-weight: 500;">{h["model_type"]}</td>
            <td style="color: #10b981; font-weight: 600;">{h["accuracy"]:.4f}</td>
            <td>{h["incumbent_accuracy"]:.4f}</td>
            <td>{h["dummy_accuracy"]:.4f}</td>
            <td style="font-size: 0.85rem; color: #94a3b8;">{h["timestamp"]}</td>
            <td><span class="status-badge {status_class}">{h["status"].upper()}</span></td>
        </tr>
        """

    # HTML UI Design aligned with Git-Ops flow
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Pipeline Promotion Gate</title>
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
            
            /* Table Styling */
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
            .status-badge.active {{
                background-color: #10b98120;
                color: #10b981;
            }}
            .status-badge.archived {{
                background-color: #64748b20;
                color: #94a3b8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Promotion Gate</h1>
            <p class="subtitle">Evaluate and promote candidate staging models to active production using Git-Ops.</p>
            
            <div class="grid">
                <!-- Incumbent Card -->
                <div class="card">
                    <span class="badge prod">PRODUCTION (Active)</span>
                    <div class="version">Model Version {active_model["version"]}</div>
                    <div class="metric-row">
                        <span class="metric-label">Model Type</span>
                        <span class="metric-value" style="text-transform: uppercase;">{active_model["model_type"]}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value" style="color: #10b981;">{active_model["accuracy"]:.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Baseline Accuracy</span>
                        <span class="metric-value">{active_model["dummy_accuracy"]:.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Max Depth</span>
                        <span class="metric-value">{active_model["parameters"].get("max_depth", "N/A")}</span>
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
                        Merging a Pull Request automatically runs the tests, checks for drift, and deploys the model live!
                    </div>
                </div>
            </div>
            
            <a href="https://github.com/ShhlokRastogi/self-healing-crypto-pipeline/pulls" target="_blank" class="button">
                Review and Merge Pull Requests on GitHub
            </a>
            
            <h2>Model Registry History</h2>
            <table>
                <thead>
                    <tr>
                        <th>Version</th>
                        <th>Type</th>
                        <th>Accuracy</th>
                        <th>Incumbent Acc.</th>
                        <th>Baseline Acc.</th>
                        <th>Registered At</th>
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

