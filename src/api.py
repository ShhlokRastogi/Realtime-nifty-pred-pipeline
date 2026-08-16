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
model_path = os.path.join(base_dir, "..", "models", "xgb_model.pkl")
model = joblib.load(model_path)
r = redis.Redis(**REDIS_CONFIG)


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
                
    # Return the metrics in the raw text format that Prometheus understands
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


from fastapi.responses import HTMLResponse

@app.get("/promote", response_class=HTMLResponse)
def promote_page():
    """Renders a beautiful model comparison page showing local model details and Git-Ops PR links."""
    import json
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(base_dir, "..", "models", "model_metadata.json")
    
    # Load metadata with a fallback for the baseline model
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        # Default baseline model metadata
        meta = {
            "version": 1,
            "accuracy": 0.5248,
            "dummy_accuracy": 0.4898,
            "timestamp": "2026-08-15 12:00:00",
            "parameters": {"max_depth": 3, "n_estimators": 69}
        }

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
                max-width: 900px;
                margin: 0 auto;
            }}
            h1 {{
                font-size: 2.2rem;
                margin-bottom: 5px;
                font-weight: 700;
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
                    <div class="version">Model Version {meta.get("version", 1)}</div>
                    <div class="metric-row">
                        <span class="metric-label">XGBoost Accuracy</span>
                        <span class="metric-value">{meta.get("accuracy", 0.5248):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Baseline Accuracy</span>
                        <span class="metric-value">{meta.get("dummy_accuracy", 0.4898):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Number of Trees</span>
                        <span class="metric-value">{meta.get("parameters", {}).get("n_estimators", "N/A")}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Max Depth</span>
                        <span class="metric-value">{meta.get("parameters", {}).get("max_depth", "N/A")}</span>
                    </div>
                    <div class="metric-row" style="border-bottom: none;">
                        <span class="metric-label">Deployed At</span>
                        <span class="metric-value" style="font-size: 0.8rem;">{meta.get("timestamp", "N/A")}</span>
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
        </div>
    </body>
    </html>
    """
    return html_content

