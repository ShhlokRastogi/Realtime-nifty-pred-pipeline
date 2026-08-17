import os
import time
import json
import joblib
import pandas as pd
import psycopg2
import optuna
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Local project imports (imported from src directory)
from config import DB_CONFIG, TICKERS, TEST_CUTOFF, RANDOM_SEED
from train import create_target_variable, split_data

# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()

FEATURE_COLS = [
    "rsi", "sma_crossover", "rolling_volatility",
    "volume_delta", "lagged_return_1", "lagged_return_3", "lagged_return_5"
]

def should_promote(candidate_accuracy: float, incumbent_accuracy: float, dummy_accuracy: float) -> bool:
    # Requires the candidate to beat the incumbent by at least 0.2% and beat the dummy baseline
    if candidate_accuracy > incumbent_accuracy + 0.002 and candidate_accuracy > dummy_accuracy:
        return True
    return False

def load_features_from_db(ticker: str) -> pd.DataFrame:
    """Read features and join with raw close prices from Postgres."""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT f.date, f.rsi, f.sma_crossover, f.rolling_volatility, 
               f.volume_delta, f.lagged_return_1, f.lagged_return_3, f.lagged_return_5,
               o.close
        FROM features f
        JOIN ohlcv o ON f.ticker = o.ticker AND f.date = o.date
        WHERE f.ticker = %s
        ORDER BY f.date ASC
    """
    df = pd.read_sql(query, conn, params=(ticker,), parse_dates=["date"])
    conn.close()
    df = df.set_index("date")
    df = df.dropna()
    return df

def optimize_xgb_hyperparameters(X_train, y_train, X_test, y_test) -> dict:
    """Uses Optuna to find the best XGBoost hyperparameters."""
    print("Running Optuna study for XGBoost...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_SEED,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    print(f"  Best XGB Accuracy: {study.best_value:.4f}")
    return study.best_params

def optimize_rf_hyperparameters(X_train, y_train, X_test, y_test) -> dict:
    """Uses Optuna to find the best Random Forest hyperparameters."""
    print("Running Optuna study for Random Forest...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": RANDOM_SEED,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    print(f"  Best RF Accuracy: {study.best_value:.4f}")
    return study.best_params

def register_model_in_db(ticker: str, model_type: str, accuracy: float, incumbent_accuracy: float, dummy_accuracy: float, parameters: dict):
    """Registers a promoted model in the Postgres database and archives the old one."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        # Set all older active models for this ticker to archived
        cur.execute("UPDATE models SET status = 'archived' WHERE ticker = %s AND status = 'active'", (ticker,))
        
        # Insert the new active model metadata
        query = """
            INSERT INTO models (ticker, model_type, accuracy, incumbent_accuracy, dummy_accuracy, parameters, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'active')
            RETURNING version;
        """
        cur.execute(query, (ticker, model_type, accuracy, incumbent_accuracy, dummy_accuracy, json.dumps(parameters)))
        version = cur.fetchone()[0]
        conn.commit()
        print(f"Successfully registered model version {version} in Postgres database!")
        return version
    except Exception as e:
        conn.rollback()
        print(f"Database error during model registration: {e}")
        raise e
    finally:
        cur.close()
        conn.close()

def get_incumbent_accuracy_from_db(ticker: str, X_test, y_test) -> float:
    """Fetches the latest active model's accuracy, or evaluates it if the file exists."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(base_dir, "..", "models", "active_model.pkl")
    
    if os.path.exists(local_model_path):
        try:
            incumbent = joblib.load(local_model_path)
            acc = accuracy_score(y_test, incumbent.predict(X_test))
            print(f"Current Incumbent (Active) model accuracy: {acc:.4f}")
            return acc
        except Exception as e:
            print(f"Could not load local active model: {e}")
    return 0.0

def run_retraining():
    ticker = TICKERS[0]
    print(f"\n=== Starting Auto-Retrain Pipeline for {ticker} ===")
    
    # Step 1: Load features
    df = load_features_from_db(ticker)
    df = create_target_variable(df)
    X_train, y_train, X_test, y_test = split_data(df)
    
    print(f"Loaded {len(df)} rows.")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Step 2: Calculate baseline dummy classifier accuracy
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Dummy Baseline Accuracy: {dummy_acc:.4f}")
    
    # Step 3: Fetch active incumbent model accuracy
    incumbent_acc = get_incumbent_accuracy_from_db(ticker, X_test, y_test)
    
    # Step 4: Run Optuna studies for both XGBoost and Random Forest
    xgb_params = optimize_xgb_hyperparameters(X_train, y_train, X_test, y_test)
    rf_params = optimize_rf_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Train both candidates to find the absolute best one
    xgb_candidate = XGBClassifier(**xgb_params, random_state=RANDOM_SEED, eval_metric="logloss")
    xgb_candidate.fit(X_train, y_train)
    xgb_candidate_acc = accuracy_score(y_test, xgb_candidate.predict(X_test))
    
    rf_candidate = RandomForestClassifier(**rf_params, random_state=RANDOM_SEED)
    rf_candidate.fit(X_train, y_train)
    rf_candidate_acc = accuracy_score(y_test, rf_candidate.predict(X_test))
    
    print(f"\nTraining complete:")
    print(f"  - XGBoost Candidate Accuracy: {xgb_candidate_acc:.4f}")
    print(f"  - Random Forest Candidate Accuracy: {rf_candidate_acc:.4f}")
    
    # Select the best model type
    if xgb_candidate_acc >= rf_candidate_acc:
        best_model = xgb_candidate
        best_type = "xgboost"
        best_acc = xgb_candidate_acc
        best_params = xgb_params
    else:
        best_model = rf_candidate
        best_type = "random_forest"
        best_acc = rf_candidate_acc
        best_params = rf_params
        
    print(f"Selected Best Candidate: {best_type.upper()} ({best_acc:.4f})")
    
    # Step 5: Evaluate the Promotion Gate
    promote = should_promote(best_acc, incumbent_acc, dummy_acc)
    print(f"\nPromotion Decision: {promote}")
    
    if promote:
        print(f"Candidate passed the gate! Saving model locally and registering...")
        
        # 1. Register in Postgres database and get Version ID
        version = register_model_in_db(ticker, best_type, best_acc, incumbent_acc, dummy_acc, best_params)
        
        # 2. Log to MLflow if tracking server is up (non-blocking)
        try:
            mlflow.set_experiment("crypto-retrain")
            with mlflow.start_run(run_name=f"{ticker}_retrain_candidate"):
                mlflow.log_params(best_params)
                mlflow.log_param("model_type", best_type)
                mlflow.log_metric("candidate_accuracy", best_acc)
                mlflow.log_metric("incumbent_accuracy", incumbent_acc)
                
                if best_type == "xgboost":
                    mlflow.xgboost.log_model(best_model, artifact_path="model", registered_model_name="crypto-model")
                else:
                    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="crypto-model")
        except Exception as e:
            print(f"MLflow Logging skipped (tracking server offline): {e}")
        
        # 3. Save model locally (Git-Ops overwrite)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(base_dir, "..", "models", "active_model.pkl")
        local_meta_path = os.path.join(base_dir, "..", "models", "model_metadata.json")
        
        joblib.dump(best_model, local_model_path)
        print(f"Saved active model weights locally to: {local_model_path}")
        
        # 4. Output metadata JSON
        metadata = {
            "version": int(version),
            "model_type": best_type,
            "accuracy": float(best_acc),
            "incumbent_accuracy": float(incumbent_acc),
            "dummy_accuracy": float(dummy_acc),
            "timestamp": str(pd.Timestamp.now()),
            "parameters": best_params
        }
        with open(local_meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved version metadata to: {local_meta_path}")
    else:
        print("Candidate failed the promotion gate. Keeping the current Production model.")

if __name__ == "__main__":
    run_retraining()