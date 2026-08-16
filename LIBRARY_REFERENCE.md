# Library Reference: Self-Healing Crypto Prediction Pipeline

This document is a technical reference guide for the third-party libraries and frameworks used in the Crypto Prediction Pipeline. It is designed to act as a copy-pasteable pattern library for future MLOps, database, and engineering work.

---

## 1. Pandas (`pandas`)
* **Purpose**: Time-series alignment, indexing, feature engineering, and preparation of tabular datasets for machine learning models.
* **Where it's used**: 
  * [`src/ingest.py`](file:///C:/D/crypto/src/ingest.py) (`fetch_ohlcv`)
  * [`src/features.py`](file:///C:/D/crypto/src/features.py) (`build_features`)
  * [`src/feature_store.py`](file:///C:/D/crypto/src/feature_store.py) (`read_ohlcv_from_db`)
  * [`src/train.py`](file:///C:/D/crypto/src/train.py) (`split_data`, `create_target_variable`)
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py) (`load_features_from_db`)
  * [`src/poll.py`](file:///C:/D/crypto/src/poll.py) (`fetch_latest_candles_binance`)
* **Core APIs used**: 
  * `pd.DataFrame()`, `pd.to_datetime()`, `pd.read_sql()`
  * `DataFrame.set_index()`, `DataFrame.dropna()`, `DataFrame.shift()`
  * `DataFrame.rolling()`, `pd.date_range()`, `DataFrame.drop()`
* **Actual code snippet** (from [`src/features.py`](file:///C:/D/crypto/src/features.py)):
  ```python
  # 1. Calculate price returns over 1, 3, and 5 periods
  df["lagged_return_1"] = df["close"].pct_change(periods=1)
  df["lagged_return_3"] = df["close"].pct_change(periods=3)
  df["lagged_return_5"] = df["close"].pct_change(periods=5)

  # 2. Moving average crossover using rolling window means
  sma_short = df["close"].rolling(window=12).mean()
  sma_long = df["close"].rolling(window=26).mean()
  df["sma_crossover"] = sma_short - sma_long

  # 3. Rolling volatility using standard deviation of returns
  df["rolling_volatility"] = df["lagged_return_1"].rolling(window=50).std()
  ```
* **Integration point**: Takes raw price candles (from APIs or Postgres), generates a processed `DataFrame` with a `DatetimeIndex`, and passes it into either the Postgres upsert query or the model training splits.
* **Reusable pattern**:
  ```python
  import pandas as pd
  
  def generate_time_series_features(raw_data_list: list) -> pd.DataFrame:
      # Convert raw JSON/list into DataFrame and set Datetime index
      df = pd.DataFrame(raw_data_list)
      df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
      df = df.set_index("timestamp").sort_index()
      
      # Causal rolling calculations
      df["feature_ma"] = df["val"].rolling(window=10).mean()
      df["feature_std"] = df["val"].rolling(window=10).std()
      
      # Drop incomplete windows to prevent passing NaNs to models
      return df.dropna()
  ```

---

## 2. Psycopg2 (`psycopg2`)
* **Purpose**: PostgreSQL database adapter used to run schemas, fetch tabular metrics, and execute high-performance bulk writes.
* **Where it's used**:
  * [`src/db_schema.py`](file:///C:/D/crypto/src/db_schema.py) (`create_tables`)
  * [`src/ingest.py`](file:///C:/D/crypto/src/ingest.py) (`save_to_postgres`)
  * [`src/feature_store.py`](file:///C:/D/crypto/src/feature_store.py) (`save_features_to_db`)
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py) (`load_features_from_db`)
* **Core APIs used**:
  * `psycopg2.connect()`, `Connection.cursor()`, `Cursor.execute()`, `Connection.commit()`
  * `psycopg2.extras.execute_values()` (used for high-performance bulk operations)
* **Actual code snippet** (from [`src/feature_store.py`](file:///C:/D/crypto/src/feature_store.py)):
  ```python
  from psycopg2.extras import execute_values

  conn = psycopg2.connect(**DB_CONFIG)
  cur = conn.cursor()

  # 1. Convert pandas rows into a flat list of Python primitive tuples
  tuples = [
      (
          ticker,
          date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
          float(row["rsi"]) if not pd.isna(row["rsi"]) else None,
          float(row["sma_crossover"]) if not pd.isna(row["sma_crossover"]) else None,
          # ... other features
      )
      for date, row in df.iterrows()
  ]

  # 2. SQL upsert query using %s placeholder for execute_values compatibility
  query = """
      INSERT INTO features (ticker, date, rsi, sma_crossover, rolling_volatility,
                            volume_delta, lagged_return_1, lagged_return_3, lagged_return_5)
      VALUES %s
      ON CONFLICT (ticker, date) DO UPDATE
      SET rsi = EXCLUDED.rsi,
          sma_crossover = EXCLUDED.sma_crossover;
  """

  # 3. Fast bulk insert - sends all 100k rows in a single batch
  execute_values(cur, query, tuples)
  conn.commit()
  ```
* **Integration point**: Interacts between Pandas (which constructs feature dataframes) and PostgreSQL tables, allowing rapid serialization/deserialization of tabular data.
* **Reusable pattern**:
  ```python
  import psycopg2
  from psycopg2.extras import execute_values
  
  def postgres_bulk_upsert(db_config: dict, table: str, columns: list, data: list):
      conn = psycopg2.connect(**db_config)
      cursor = conn.cursor()
      
      col_names = ", ".join(columns)
      value_placeholders = "%s"
      query = f"INSERT INTO {table} ({col_names}) VALUES {value_placeholders}"
      
      # execute_values dynamically compiles the list of tuples into a multi-row insert statement
      execute_values(cursor, query, data)
      conn.commit()
      cursor.close()
      conn.close()
  ```

---

## 3. Redis (`redis`)
* **Purpose**: High-speed in-memory database used to cache the latest feature record, allowing the FastAPI model serving endpoint to retrieve features in under 1 millisecond.
* **Where it's used**:
  * [`src/feature_store.py`](file:///C:/D/crypto/src/feature_store.py) (`cache_latest_in_redis`)
  * [`src/api.py`](file:///C:/D/crypto/src/api.py) (`predict`)
* **Core APIs used**:
  * `redis.Redis()`, `Redis.set()`, `Redis.get()`
* **Actual code snippet** (from [`src/feature_store.py`](file:///C:/D/crypto/src/feature_store.py) and [`src/api.py`](file:///C:/D/crypto/src/api.py)):
  ```python
  # --- Writing to Cache (feature_store.py) ---
  r = redis.Redis(**REDIS_CONFIG)
  latest = df.iloc[-1]  # Get most recent feature row
  
  payload = {
      "ticker": ticker,
      "date": str(latest.name),
      "features": {col: float(latest[col]) for col in feature_cols}
  }
  
  # Serialize payload to JSON string and store
  r.set(f"features:{ticker}", json.dumps(payload))

  # --- Reading from Cache (api.py) ---
  raw = r.get(f"features:{ticker}")
  if raw is not None:
      data = json.loads(raw)
      features = data["features"]
  ```
* **Integration point**: Bridges the feature computation script (`feature_store.py` running in the background) and the live FastAPI web server, removing the need for the web server to query the heavy PostgreSQL database during prediction requests.
* **Reusable pattern**:
  ```python
  import redis
  import json
  
  class RedisCache:
      def __init__(self, host='localhost', port=6379, db=0):
          self.client = redis.Redis(host=host, port=port, db=db)
          
      def set_json(self, key: str, data: dict):
          self.client.set(key, json.dumps(data))
          
      def get_json(self, key: str) -> dict:
          raw = self.client.get(key)
          return json.loads(raw) if raw else None
  ```

---

## 4. FastAPI (`fastapi`)
* **Purpose**: Microservice framework used to host prediction endpoints, expose Prometheus metrics, and serve the model promotion web interface.
* **Where it's used**:
  * [`src/api.py`](file:///C:/D/crypto/src/api.py) (Entire file)
* **Core APIs used**:
  * `FastAPI()`, `HTTPException`, `Response`, `HTMLResponse`
  * Route decorators: `@app.get()`, `@app.post()`
* **Actual code snippet** (from [`src/api.py`](file:///C:/D/crypto/src/api.py)):
  ```python
  from fastapi import FastAPI, HTTPException, Response
  from fastapi.responses import HTMLResponse

  app = FastAPI(title="Crypto Price Prediction API")

  @app.get("/predict/{ticker}")
  def predict(ticker: str):
      raw = r.get(f"features:{ticker}")
      if raw is None:
          # Raise standard HTTP 404 error if key doesn't exist
          raise HTTPException(status_code=404, detail="Features not found")
      
      # Process features and run prediction...
      prediction = model.predict(values)[0]
      return {"ticker": ticker, "prediction": int(prediction)}

  @app.post("/promote/approve")
  def approve_promotion():
      # Handles state transitions and reloads the global model in memory
      global model
      model = mlflow.xgboost.load_model("models:/crypto-model/Production")
      return HTMLResponse(content="<h1>Promotion Successful!</h1>")
  ```
* **Integration point**: Serves as the central API gateway. It takes JSON requests from clients, loads models from MLflow, reads features from Redis, increments Prometheus counters, and outputs prediction signals.
* **Reusable pattern**:
  ```python
  from fastapi import FastAPI, HTTPException
  
  app = FastAPI(title="MLOps Microservice")
  
  @app.get("/health")
  def health_check():
      return {"status": "healthy"}
      
  @app.post("/predict")
  def run_prediction(payload: dict):
      try:
          # Run classification model logic
          return {"prediction": 1}
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
  ```

---

## 5. Prometheus Client (`prometheus_client`)
* **Purpose**: Exposes internal system metrics (request counts, latency) and statistical ML metrics (drift status, KS-test p-values) in a text format that the Prometheus scraper understands.
* **Where it's used**:
  * [`src/api.py`](file:///C:/D/crypto/src/api.py)
* **Core APIs used**:
  * `REGISTRY`, `Counter()`, `Gauge()`, `generate_latest()`, `CONTENT_TYPE_LATEST`
* **Actual code snippet** (from [`src/api.py`](file:///C:/D/crypto/src/api.py)):
  ```python
  from prometheus_client import REGISTRY, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

  # 1. Prevent duplicate metric crashes during FastAPI/Uvicorn hot-reloads
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

  # 2. Expose the metrics endpoint for the Prometheus scraper
  @app.get("/metrics")
  def metrics():
      for t in TICKERS:
          drift_detected = 1.0 if check_drift(t) else 0.0
          DRIFT_GAUGE.labels(ticker=t).set(drift_detected)
          
      # generate_latest() compiles all metrics into Prometheus plain text format
      return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
  ```
* **Integration point**: Converts internal Python model metrics into a standard scraper endpoint (`/metrics`) which is queried by the Prometheus Docker container every 5 seconds.
* **Reusable pattern**:
  ```python
  from fastapi import FastAPI, Response
  from prometheus_client import REGISTRY, Counter, generate_latest, CONTENT_TYPE_LATEST
  
  app = FastAPI()
  
  # Register counter safely
  if "api_calls_total" not in REGISTRY._names_to_collectors:
      CALL_COUNTER = Counter("api_calls_total", "Number of total API calls", ["endpoint"])
  else:
      CALL_COUNTER = REGISTRY._names_to_collectors["api_calls_total"]
  
  @app.get("/metrics")
  def get_metrics():
      return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
  ```

---

## 6. MLflow (`mlflow`)
* **Purpose**: Manages model tracking (parameters, metrics, run files) and hosts the Model Registry to manage "Staging" and "Production" model transitions.
* **Where it's used**:
  * [`src/train.py`](file:///C:/D/crypto/src/train.py) (`log_model`)
  * [`src/api.py`](file:///C:/D/crypto/src/api.py) (`load_model`)
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py) (`log_model`, `transition_model_version_stage`)
* **Core APIs used**:
  * `mlflow.set_tracking_uri()`, `mlflow.set_experiment()`, `mlflow.start_run()`
  * `mlflow.log_param()`, `mlflow.log_metric()`
  * `mlflow.xgboost.log_model()`, `mlflow.xgboost.load_model()`
  * `MlflowClient`, `client.transition_model_version_stage()`
* **Actual code snippet** (from [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py)):
  ```python
  import mlflow
  import mlflow.xgboost
  from mlflow.tracking import MlflowClient

  # 1. Connect to local SQLite-backed MLflow server
  mlflow.set_tracking_uri("http://127.0.0.1:5000")
  client = MlflowClient()

  with mlflow.start_run(run_name="retrain_candidate"):
      mlflow.log_param("model_type", "XGBoost")
      mlflow.log_metric("accuracy", candidate_acc)
      
      # 2. Save model and register it in Model Registry (creates Version 2)
      model_info = mlflow.xgboost.log_model(
          candidate_model, 
          artifact_path="model", 
          registered_model_name="crypto-model"
      )
      
      # 3. Transition the new version to "Staging"
      client.transition_model_version_stage(
          name="crypto-model",
          version=model_info.registered_model_version,
          stage="Staging"
      )
  ```
* **Integration point**: Serves as the central repository for models. FastAPI queries it to load active models, and retraining scripts push candidate versions to it.
* **Reusable pattern**:
  ```python
  import mlflow
  
  class MLflowRegistryHelper:
      def __init__(self, tracking_uri="http://localhost:5000"):
          mlflow.set_tracking_uri(tracking_uri)
          
      def log_and_register(self, model, experiment: str, run_name: str, model_name: str, metrics: dict):
          mlflow.set_experiment(experiment)
          with mlflow.start_run(run_name=run_name):
              for k, v in metrics.items():
                  mlflow.log_metric(k, v)
              # Returns details of registered model (including version)
              return mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
  ```

---

## 7. Optuna (`optuna`)
* **Purpose**: Automates the search for optimal hyperparameters for ML models using Bayesian optimization.
* **Where it's used**:
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py) (`optimize_hyperparameters`)
* **Core APIs used**:
  * `optuna.create_study()`, `Study.optimize()`
  * `Trial.suggest_int()`, `Trial.suggest_float()`
* **Actual code snippet** (from [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py)):
  ```python
  import optuna

  def optimize_hyperparameters(X_train, y_train, X_test, y_test) -> dict:
      # Objective function defines the search space and evaluation metric
      def objective(trial):
          params = {
              "n_estimators": trial.suggest_int("n_estimators", 50, 200),
              "max_depth": trial.suggest_int("max_depth", 3, 9),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
              "subsample": trial.suggest_float("subsample", 0.6, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
              "random_state": 42,
              "eval_metric": "logloss",
          }
          
          model = XGBClassifier(**params)
          model.fit(X_train, y_train)
          preds = model.predict(X_test)
          return accuracy_score(y_test, preds)

      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=20) # Executes 20 training trials
      return study.best_params
  ```
* **Integration point**: Runs inside the retraining pipeline. It ingests training sets, iteratively runs trials, and outputs the optimal parameter dictionary directly to the candidate model initializer.
* **Reusable pattern**:
  ```python
  import optuna
  
  def tune_model(X_train, y_train, X_val, y_val, metric_fn):
      def objective(trial):
          # Define hyperparameter suggestions
          param_val = trial.suggest_int("param_name", 10, 100)
          # Train and validate model...
          metric = metric_fn(y_val, preds)
          return metric
          
      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=10)
      return study.best_params
  ```

---

## 8. XGBoost (`xgboost`)
* **Purpose**: Extreme Gradient Boosting decision tree framework used to train the directional price prediction model.
* **Where it's used**:
  * [`src/train.py`](file:///C:/D/crypto/src/train.py) (`train_model`)
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py)
* **Core APIs used**:
  * `XGBClassifier()`, `XGBClassifier.fit()`, `XGBClassifier.predict()`, `XGBClassifier.predict_proba()`
* **Actual code snippet** (from [`src/train.py`](file:///C:/D/crypto/src/train.py) and [`src/api.py`](file:///C:/D/crypto/src/api.py)):
  ```python
  # --- Training the Model (train.py) ---
  # Initialize with logloss metrics to avoid warnings
  model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
  model.fit(X_train, y_train)

  # --- Querying the Model (api.py) ---
  # Returns binary array prediction: [1] (UP) or [0] (DOWN)
  prediction = model.predict(values)[0]
  # Returns probability arrays: [P(0), P(1)]
  probabilities = model.predict_proba(values)[0]
  confidence = float(max(probabilities))
  ```
* **Integration point**: Fits on training dataframes, and is loaded as a serialized object inside FastAPI to execute predictions on live inputs.
* **Reusable pattern**:
  ```python
  from xgboost import XGBClassifier
  
  def train_and_predict(X_train, y_train, X_new):
      model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss")
      model.fit(X_train, y_train)
      
      prediction = model.predict(X_new)[0]
      confidence = model.predict_proba(X_new)[0][prediction]
      return prediction, confidence
  ```

---

## 9. Scikit-Learn (`sklearn`)
* **Purpose**: Used to train a dummy baseline model (majority class classifier) and generate standard metrics to verify if our ML model is actually learning.
* **Where it's used**:
  * [`src/train.py`](file:///C:/D/crypto/src/train.py) (`evaluate_model`)
  * [`src/retrain.py`](file:///C:/D/crypto/src/retrain.py)
* **Core APIs used**:
  * `DummyClassifier()`, `accuracy_score()`, `classification_report()`, `confusion_matrix()`
* **Actual code snippet** (from [`src/train.py`](file:///C:/D/crypto/src/train.py)):
  ```python
  from sklearn.dummy import DummyClassifier
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

  # 1. Train baseline Dummy model
  dummy = DummyClassifier(strategy="most_frequent", random_state=42)
  dummy.fit(X_train, y_train)

  # 2. Evaluate metrics
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  
  # 3. Print out descriptive matrix reports
  print(classification_report(y_test, y_pred, zero_division=0))
  print(confusion_matrix(y_test, y_pred))
  ```
* **Integration point**: Computes mathematical validation scores comparing candidate models, incumbent models, and basic statistical dummies.
* **Reusable pattern**:
  ```python
  from sklearn.metrics import accuracy_score, classification_report
  
  def validate_predictions(y_true, y_pred):
      acc = accuracy_score(y_true, y_pred)
      report = classification_report(y_true, y_pred, output_dict=True)
      return acc, report
  ```

---

## 10. Requests (`requests`)
* **Purpose**: Executes HTTP calls to the public Binance REST API to fetch raw historical candles and query live streaming updates.
* **Where it's used**:
  * [`src/ingest.py`](file:///C:/D/crypto/src/ingest.py) (`fetch_ohlcv`)
  * [`src/poll.py`](file:///C:/D/crypto/src/poll.py) (`fetch_latest_candles_binance`)
* **Core APIs used**:
  * `requests.get()`, `Response.json()`, `Response.raise_for_status()`
* **Actual code snippet** (from [`src/ingest.py`](file:///C:/D/crypto/src/ingest.py)):
  ```python
  import requests

  url = "https://api.binance.com/api/v3/klines"
  params = {
      "symbol": symbol,
      "interval": "15m",
      "startTime": current_start,
      "endTime": end_time,
      "limit": 1000
  }

  # Execute HTTP GET request
  response = requests.get(url, params=params)
  # Throw an HTTPError if the response code was an error (e.g. 400, 429)
  response.raise_for_status()
  data = response.json()
  ```
* **Integration point**: Queries raw pricing metrics from exchange API endpoints, converts the output JSON arrays into Python dictionaries, and passes them to Pandas.
* **Reusable pattern**:
  ```python
  import requests
  
  def query_json_api(endpoint: str, params: dict) -> dict:
      response = requests.get(endpoint, params=params)
      response.raise_for_status()
      return response.json()
  ```

---

## How It All Fits Together

Here is the end-to-end data lifecycle of the pipeline, showing which library governs each stage:

```
                                  [ Binance Exchange REST API ]
                                                │
                                        (HTTP requests)
                                                ▼
                                   [ Ingestion & Polling ]
                                      (Library: requests)
                                                │
                                                ▼
                                       [ Data Preparation ]
                                        (Library: pandas)
                                                │
                                                ▼
                                         [ Storage Layer ]
                                       (Library: psycopg2)
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
                 [ Postgres Tables ]                         [ Postgres Tables ]
                  (Raw OHLCV data)                           (Calculated Features)
                          │                                           │
                          │                                           ▼
                          │                              [ Feature Store Pipeline ]
                          │                                 (Library: psycopg2)
                          │                                           │
                          │                          ┌────────────────┴────────────────┐
                          │                          ▼                                 ▼
                          │                 [ Postgres Tables ]                 [ Redis Cache ]
                          │                 (Computed Features)              (Library: redis / json)
                          │                          │                                 │
                          │                          │                           (Cache reads)
                          │                          ▼                                 ▼
                  [ train.py / retrain.py ] <────────┘                         [ fastapi Service ]
                  (Library: sklearn / optuna)                                 (Library: fastapi)
                          │                                                            ▲
                   (Trains models)                                                     │
                          ▼                                                     (Loads active)
                  [ Model Registry ] ──────────────────────────────────────────┘
             (Library: mlflow / sqlite)
                          │
                          ▼
                  [ prometheus_client ]
                  (Exposes `/metrics`)
                          │
                          ▼
                  [ Grafana Dashboard ]
```
