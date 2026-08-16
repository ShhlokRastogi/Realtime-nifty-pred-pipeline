import json
import joblib
import redis
from fastapi import FastAPI, HTTPException, Response
from config import REDIS_CONFIG, TICKERS
from prometheus_client import REGISTRY, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from drift import run_drift_check, FEATURE_COLS

app = FastAPI(title="Crypto Price Prediction API")
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.xgboost.load_model("models:/crypto-model/Production")
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
from mlflow.tracking import MlflowClient

@app.get("/promote", response_class=HTMLResponse)
def promote_page():
    """Renders a beautiful model comparison page to approve staging models."""
    client = MlflowClient()
    
    prod_version = None
    stage_version = None
    prod_metrics = {}
    stage_metrics = {}
    prod_params = {}
    stage_params = {}
    
    # 1. Search for registered versions of the model
    versions = client.search_model_versions("name='crypto-model'")
    for v in versions:
        if v.current_stage == "Production":
            prod_version = v
            run = client.get_run(v.run_id)
            prod_metrics = run.data.metrics
            prod_params = run.data.params
        elif v.current_stage == "Staging":
            stage_version = v
            run = client.get_run(v.run_id)
            stage_metrics = run.data.metrics
            stage_params = run.data.params

    # HTML UI Design with clean, modern CSS styling
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
                transition: background-color 0.2s;
            }}
            .button:hover {{
                background-color: #2563eb;
            }}
            .button:disabled {{
                background-color: #475569;
                color: #94a3b8;
                cursor: not-allowed;
            }}
            .no-model {{
                color: #64748b;
                text-align: center;
                padding: 40px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Promotion Gate</h1>
            <p class="subtitle">Evaluate and promote candidate staging models to active production.</p>
            
            <div class="grid">
                <!-- Incumbent Card -->
                <div class="card">
                    <span class="badge prod">PRODUCTION (Active)</span>
                    {f'''
                    <div class="version">Model Version {prod_version.version}</div>
                    <div class="metric-row">
                        <span class="metric-label">XGBoost Accuracy</span>
                        <span class="metric-value">{prod_metrics.get("xgb_accuracy", 0.0):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Dummy Baseline Accuracy</span>
                        <span class="metric-value">{prod_metrics.get("dummy_accuracy", 0.0):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Number of Trees</span>
                        <span class="metric-value">{prod_params.get("n_estimators", "N/A")}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Max Depth</span>
                        <span class="metric-value">{prod_params.get("max_depth", "N/A")}</span>
                    </div>
                    ''' if prod_version else '<div class="no-model">No active model in Production</div>'}
                </div>
                
                <!-- Candidate Card -->
                <div class="card staging">
                    <span class="badge stage">STAGING (Candidate)</span>
                    {f'''
                    <div class="version">Model Version {stage_version.version}</div>
                    <div class="metric-row">
                        <span class="metric-label">Candidate Accuracy</span>
                        <span class="metric-value">{stage_metrics.get("candidate_accuracy", 0.0):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Dummy Baseline Accuracy</span>
                        <span class="metric-value">{stage_metrics.get("dummy_accuracy", 0.0):.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Number of Trees</span>
                        <span class="metric-value">{stage_params.get("n_estimators", "N/A")}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Max Depth</span>
                        <span class="metric-value">{stage_params.get("max_depth", "N/A")}</span>
                    </div>
                    ''' if stage_version else '<div class="no-model">No candidate model in Staging</div>'}
                </div>
            </div>
            
            <form action="/promote/approve" method="POST">
                <button type="submit" class="button" {"" if stage_version else "disabled"}>
                    Approve and Promote to Production (Hot-Swap)
                </button>
            </form>
        </div>
    </body>
    </html>
    """
    return html_content


@app.post("/promote/approve")
def approve_promotion():
    """Handles the promotion, archiving the old model and hot-swapping the active model."""
    global model
    client = MlflowClient()
    
    prod_version = None
    stage_version = None
    
    # Find current versions
    versions = client.search_model_versions("name='crypto-model'")
    for v in versions:
        if v.current_stage == "Production":
            prod_version = v
        elif v.current_stage == "Staging":
            stage_version = v

    if not stage_version:
        raise HTTPException(status_code=400, detail="No candidate model found in Staging.")

    # 1. Demote current Production model to None/Archived
    if prod_version:
        client.transition_model_version_stage(
            name="crypto-model",
            version=prod_version.version,
            stage="Archived"
        )

    # 2. Promote Staging model to Production
    client.transition_model_version_stage(
        name="crypto-model",
        version=stage_version.version,
        stage="Production"
    )

    # 3. HOT-SWAP: Reload the new model instantly in memory
    print(f"Hot-swapping active model to Version {stage_version.version}...")
    model = mlflow.xgboost.load_model("models:/crypto-model/Production")
    
    # Return HTML success page
    html_success = f"""
    <html>
    <body style="font-family: sans-serif; background-color: #0f172a; color: white; text-align: center; padding-top: 100px;">
        <h1 style="color: #10b981;">Promotion Successful!</h1>
        <p>Model version <b>{stage_version.version}</b> is now in <b>Production</b> and running live.</p>
        <a href="/promote" style="color: #3b82f6; text-decoration: none;">Go back to Dashboard</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_success)

