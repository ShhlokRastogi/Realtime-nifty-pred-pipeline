import os
import time
import pandas as pd
import psycopg2
import optuna
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Local project imports (imported from src directory)
from config import DB_CONFIG, TICKERS, TEST_CUTOFF, RANDOM_SEED
from features import build_features
from train import create_target_variable, split_data


# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

def should_promote(candidate_accuracy: float, incumbent_accuracy: float, dummy_accuracy: float) -> bool:
    # Requires the candidate to beat the incumbent by at least 0.2% and beat the dummy baseline
    if(candidate_accuracy > incumbent_accuracy + 0.002 and candidate_accuracy > dummy_accuracy):
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

def optimize_hyperparameters(X_train, y_train, X_test, y_test) -> dict:
    """Uses Optuna to find the best XGBoost hyperparameters."""
    print("Running Optuna study...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_SEED,
            "eval_metric": "logloss",
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print(f"  Best Accuracy found by Optuna: {study.best_value:.4f}")
    return study.best_params

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
    # Step 3: Fetch active "Production" model (incumbent) from MLflow
    incumbent_acc = 0.0
    try:
        incumbent = mlflow.xgboost.load_model("models:/crypto-model/Production")
        incumbent_acc = accuracy_score(y_test, incumbent.predict(X_test))
        print(f"Current Incumbent (Production) model accuracy: {incumbent_acc:.4f}")
    except Exception as e:
        print("No active Production model found in MLflow.")
    # Step 4: Run Optuna to find best hyperparameters
    best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
    print(f"Best parameters: {best_params}")
    # Step 5: Train candidate model using best hyperparameters
    print("Training candidate model...")
    candidate_model = XGBClassifier(**best_params, random_state=RANDOM_SEED, eval_metric="logloss")
    candidate_model.fit(X_train, y_train)
    candidate_acc = accuracy_score(y_test, candidate_model.predict(X_test))
    print(f"Candidate model accuracy: {candidate_acc:.4f}")
    # Step 6: Log Candidate to MLflow and check Promotion Gate
    mlflow.set_experiment("crypto-retrain")
    with mlflow.start_run(run_name=f"{ticker}_retrain_candidate"):
        mlflow.log_params(best_params)
        mlflow.log_metric("candidate_accuracy", candidate_acc)
        mlflow.log_metric("incumbent_accuracy", incumbent_acc)
        mlflow.log_metric("dummy_accuracy", dummy_acc)
        mlflow.log_metric("candidate_vs_incumbent", candidate_acc - incumbent_acc)
        # Evaluate the Promotion Gate
        promote = should_promote(candidate_acc, incumbent_acc, dummy_acc)
        print(f"\nPromotion Decision: {promote}")
        if promote:
            print("Candidate passed the gate! Saving model locally and registering in MLflow Staging...")
            # 1. Log to MLflow
            model_info = mlflow.xgboost.log_model(
                candidate_model, 
                artifact_path="model", 
                registered_model_name="crypto-model"
            )
            version = model_info.registered_model_version
            client.transition_model_version_stage(
                name="crypto-model",
                version=version,
                stage="Staging"
            )
            print(f"Successfully registered model version {version} in stage 'Staging'!")

            # 2. Save model locally (Git-Ops overwrite)
            import joblib
            import json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_model_path = os.path.join(base_dir, "..", "models", "xgb_model.pkl")
            local_meta_path = os.path.join(base_dir, "..", "models", "model_metadata.json")

            joblib.dump(candidate_model, local_model_path)
            print(f"Saved candidate model locally to: {local_model_path}")

            # 3. Output metadata JSON
            metadata = {
                "version": int(version),
                "accuracy": float(candidate_acc),
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