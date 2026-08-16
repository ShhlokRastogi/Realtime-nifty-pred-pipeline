import os
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from features import build_features
from config import (
    TEST_CUTOFF, RANDOM_SEED, RAW_DATA_DIR, TICKERS,
    RSI_PERIOD, SMA_SHORT, SMA_LONG, VOLATILITY_WINDOW,
    VOLUME_DELTA_WINDOW, LAGGED_RETURN_PERIODS
)


# Columns that are raw OHLCV data — not useful as model features
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target variable: 1 if next-day close > today's close, else 0.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' prices.

    Returns:
    pd.DataFrame: DataFrame with the 'target' column added. Last row dropped
                  (no "tomorrow" to compare against).
    """
    tomorrow_close = df['close'].shift(-1)
    df['target'] = (tomorrow_close > df['close']).astype(int)
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split the DataFrame into training and testing sets using a date cutoff.
    Uses TEST_CUTOFF from config — everything before = train, on/after = test.
    No random shuffling (that would leak future data).

    Parameters:
    df (pd.DataFrame): DataFrame with features and 'target' column.

    Returns:
    tuple: (X_train, y_train, X_test, y_test)
    """
    # Split by date
    train_df = df[df.index < TEST_CUTOFF]
    test_df = df[df.index >= TEST_CUTOFF]

    # Separate features (X) from label (y), drop raw OHLCV columns
    cols_to_drop = ['target'] + [c for c in OHLCV_COLS if c in df.columns]

    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['target']

    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['target']

    print(f"Train: {len(X_train)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"Test:  {len(X_test)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    print(f"Features: {list(X_train.columns)}")

    return X_train, y_train, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.

    Returns:
    XGBClassifier: Trained model.
    """
    model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate a trained model and print performance metrics.

    Parameters:
    model: Trained classifier (XGBoost, Dummy, etc.)
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing target variable.

    Returns:
    float: Accuracy score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy


if __name__ == "__main__":
    # Load raw data for the first ticker (BTC-USD)
    ticker = TICKERS[0]
    print(f"=== Training on {ticker} ===\n")

    df = pd.read_csv(f"{RAW_DATA_DIR}/{ticker}.csv", index_col=0, parse_dates=True)
    df = build_features(df)
    df = create_target_variable(df)

    X_train, y_train, X_test, y_test = split_data(df)

    # ── MLflow: set up experiment ────────────────────────
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("crypto-baseline")

    with mlflow.start_run(run_name=f"{ticker}_baseline_v1"):

        # Log parameters (so we can reproduce this exact run later)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("test_cutoff", TEST_CUTOFF)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", str(list(X_train.columns)))
        mlflow.log_param("rsi_period", RSI_PERIOD)
        mlflow.log_param("sma_short", SMA_SHORT)
        mlflow.log_param("sma_long", SMA_LONG)
        mlflow.log_param("volatility_window", VOLATILITY_WINDOW)
        mlflow.log_param("volume_delta_window", VOLUME_DELTA_WINDOW)
        mlflow.log_param("lagged_return_periods", str(LAGGED_RETURN_PERIODS))

        # --- Dummy Classifier (the floor) ---
        print("\n=== Dummy Classifier (majority class) ===")
        dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
        dummy.fit(X_train, y_train)
        dummy_acc = evaluate_model(dummy, X_test, y_test)
        mlflow.log_metric("dummy_accuracy", dummy_acc)

        # --- XGBoost ---
        print("\n=== XGBoost ===")
        xgb_model = train_model(X_train, y_train)
        xgb_acc = evaluate_model(xgb_model, X_test, y_test)
        mlflow.log_metric("xgb_accuracy", xgb_acc)
        mlflow.log_metric("xgb_vs_dummy", xgb_acc - dummy_acc)

        # Save the trained model as an MLflow artifact and register it
        mlflow.xgboost.log_model(xgb_model, artifact_path="model", registered_model_name="crypto-model")

        # --- Summary ---
        print("\n=== Summary ===")
        print(f"Dummy accuracy:   {dummy_acc:.4f}")
        print(f"XGBoost accuracy: {xgb_acc:.4f}")
        print(f"XGBoost beats dummy by: {xgb_acc - dummy_acc:+.4f}")
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print("View runs:  mlflow ui  (then open http://localhost:5000)")

        # Save model to disk for the FastAPI endpoint
        os.makedirs("models", exist_ok=True)
        joblib.dump(xgb_model, "models/xgb_model.pkl")
        print("Model saved to models/xgb_model.pkl")