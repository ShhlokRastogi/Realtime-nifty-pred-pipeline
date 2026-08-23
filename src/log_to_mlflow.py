import os
import json
import torch
import mlflow
import mlflow.pytorch
from train_reg import AttentionGRURegressor
from config import SEQ_LEN, VOL_FORECAST_WINDOW, LOOKBACK_SIZE

def log_to_mlflow_registry():
    print("Starting MLflow logging process...")
    
    # Configure MLflow target
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Nifty_Volatility_Regression")
    
    # Path validations
    model_weights_path = "models/attention_regressor.pt"
    metrics_path = "models/metrics.json"
    curves_path = "models/training_curves.png"
    
    if not all(os.path.exists(p) for p in [model_weights_path, metrics_path, curves_path]):
        raise FileNotFoundError("Local model, metrics, or curves missing in models/. Run train_reg.py first.")
        
    # 1. Load the locally saved model weights
    model = AttentionGRURegressor(input_dim=16, hidden_dim=256, num_layers=3, dropout=0.19345)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 2. Read the locally saved metrics JSON
    with open(metrics_path, "r") as f_in:
        metrics = json.load(f_in)
        
    # 3. Log to MLflow Run
    with mlflow.start_run() as run:
        print(f"Logged under Run ID: {run.info.run_id}")
        
        # Log Hyperparameters
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "forecast_window": VOL_FORECAST_WINDOW,
            "lookback_size": LOOKBACK_SIZE,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.19345,
            "lr": 0.00025894,
            "optimizer": "AdamW"
        })
        
        # Log Backtest Metrics
        mlflow.log_metrics({
            "backtest_mae": metrics["mae"],
            "backtest_r2": metrics["r2"],
            "backtest_directional_accuracy": metrics["dir_accuracy"]
        })
        
        # Log training curves PNG
        mlflow.log_artifact(curves_path)
        
        # Log and Register model
        print("Registering PyTorch model to the MLflow Registry...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="NiftyVolatilityRegressor",
            serialization_format="pickle" # Force standard state-dict pickle format
        )
        
    print("Successfully logged run metrics and registered model in MLflow!")

if __name__ == "__main__":
    log_to_mlflow_registry()