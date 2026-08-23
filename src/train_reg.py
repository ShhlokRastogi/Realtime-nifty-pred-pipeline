import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import copy
from config import PROCESSED_DATA_DIR, VOL_FORECAST_WINDOW, SEQ_LEN, LOOKBACK_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Column Definitions
FEATURE_COLS_VOL = [
    "rsi", "macd_diff_pct", "bb_width", "atr_pct", "hl_spread", "volume_delta", 
    "lagged_return_1", "vix", "vix_return", "realized_vol_5", "realized_vol_10", 
    "realized_vol_20", "sin_hour", "cos_hour", "sin_day", "cos_day"
]

# Model definitions
class TemporalPriorAttention4h(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int = 42, bias_len: int = 6, bias_weight: float = 2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, 1))
        bias = torch.zeros(seq_len)
        bias[-bias_len:] = bias_weight
        self.bias = nn.Parameter(bias.unsqueeze(0), requires_grad=False)

    def forward(self, gru_out):
        raw_scores = torch.matmul(gru_out, self.weight)
        scores = raw_scores.squeeze(-1) + self.bias
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(gru_out * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionGRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalPriorAttention4h(hidden_dim, seq_len=42, bias_len=6, bias_weight=2.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, hn = self.gru(x)
        context, weights = self.attention(out)
        prediction = self.fc(context)
        return prediction.squeeze(-1)

# Upgraded training helper to track and print epoch data
def train_champion_regressor(X_tr, y_tr, epochs=30, batch_size=32, verbose=False):
    scaler_local = StandardScaler()
    N_tr, S, F = X_tr.shape
    X_tr_scaled = scaler_local.fit_transform(X_tr.reshape(-1, F)).reshape(N_tr, S, F)
    
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr_scaled), torch.FloatTensor(y_tr)), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    model = AttentionGRURegressor(
        input_dim=16, hidden_dim=256, num_layers=3, dropout=0.19345
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025894, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    epoch_losses = []
    lrs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            
        epoch_loss /= len(loader.dataset)
        scheduler.step(epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_losses.append(epoch_loss)
        lrs.append(current_lr)
        
        if verbose:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.6f} | LR: {current_lr:.8f}")
            
    return model, scaler_local, epoch_losses, lrs

def run_regression_pipeline():
    processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_features.csv")
    df = pd.read_csv(processed_path, index_col='Datetime', parse_dates=True)
    
    X_raw_continuous = df[FEATURE_COLS_VOL].values
    closes = df['close'].values
    realized_vol_5_raw = df['realized_vol_5'].values
    
    X_seq_list, y_seq_list = [], []
    for i in range(len(df) - SEQ_LEN - VOL_FORECAST_WINDOW):
        seq_x = X_raw_continuous[i : i + SEQ_LEN]
        future_vols = realized_vol_5_raw[i + SEQ_LEN - 1 + 1 : i + SEQ_LEN - 1 + 1 + VOL_FORECAST_WINDOW]
        avg_future_vol = np.mean(future_vols)
        scaled_target = avg_future_vol * 100.0
        X_seq_list.append(seq_x)
        y_seq_list.append(scaled_target)
        
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)
    
    split_idx = int(len(X_seq) * 0.8)
    current_idx = split_idx
    total_len = len(X_seq)
    window_size = 60
    
    walk_forward_preds = []
    walk_forward_actuals = []
    
    # A. Backtest run (verbose=False to avoid massive terminal scrolling)
    print(f"Starting Volatility walk-forward backtest (Predicting {split_idx} to {total_len})...")
    while current_idx < total_len:
        end_idx = min(current_idx + window_size, total_len)
        X_tr_slice = X_seq[current_idx - LOOKBACK_SIZE : current_idx]
        y_tr_slice = y_seq[current_idx - LOOKBACK_SIZE : current_idx]
        
        # Train silently
        model, scaler_local, _, _ = train_champion_regressor(X_tr_slice, y_tr_slice, verbose=False)
        
        X_ts_slice = X_seq[current_idx:end_idx]
        y_ts_actual = y_seq[current_idx:end_idx]
        N_ts, S, F = X_ts_slice.shape
        X_ts_scaled = scaler_local.transform(X_ts_slice.reshape(-1, F)).reshape(N_ts, S, F)
        
        model.eval()
        with torch.no_grad():
            X_ts_tensor = torch.FloatTensor(X_ts_scaled).to(device)
            preds = model(X_ts_tensor).cpu().numpy()
            
        walk_forward_preds.extend(preds)
        walk_forward_actuals.extend(y_ts_actual)
        current_idx = end_idx
        
    wf_preds = np.array(walk_forward_preds)
    wf_actuals = np.array(walk_forward_actuals)
    
    mae = mean_absolute_error(wf_actuals, wf_preds)
    r2 = r2_score(wf_actuals, wf_preds)
    
    current_vol_baseline = realized_vol_5_raw[split_idx : split_idx + len(wf_preds)] * 100.0
    pred_change = np.where(wf_preds > current_vol_baseline, 1, 0)
    actual_change = np.where(wf_actuals > current_vol_baseline, 1, 0)
    dir_accuracy = accuracy_score(actual_change, pred_change) * 100
    
    print("\n" + "="*20 + " BACKTEST METRICS " + "="*20)
    print(f"MAE  : {mae:.4f}%")
    print(f"R2   : {r2*100:.2f}%")
    print(f"Dir  : {dir_accuracy:.2f}%")
    print("="*58)
    
    # B. Final model training run (verbose=True + save curves!)
    print("\nTraining final production model (with verbose logs)...")
    final_X_tr = X_seq[total_len - LOOKBACK_SIZE : total_len]
    final_y_tr = y_seq[total_len - LOOKBACK_SIZE : total_len]
    
    production_model, production_scaler, losses, lrs = train_champion_regressor(
        final_X_tr, final_y_tr, epochs=30, batch_size=32, verbose=True
    )
    
    # Save the final model state
    os.makedirs("models", exist_ok=True)
    torch.save(production_model.state_dict(), "models/attention_regressor.pt")
    with open("models/scaler_regressor.pkl", "wb") as f_out:
        pickle.dump(production_scaler, f_out)
    
    # Save metrics JSON locally (So MLflow log script can read them)
    metrics = {
        "mae": float(mae),
        "r2": float(r2 * 100.0),
        "dir_accuracy": float(dir_accuracy)
    }
    with open("models/metrics.json", "w") as f_met:
        json.dump(metrics, f_met)
    
    # C. Generate and save the training curves plot
    print("Generating training curves plot...")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot loss on left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss', color='tab:red')
    ax1.plot(range(1, len(losses)+1), losses, color='tab:red', linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate on right y-axis (twin axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue')
    ax2.plot(range(1, len(lrs)+1), lrs, color='tab:blue', linestyle='--', linewidth=1.8, label='LR')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Production Volatility Model Training History')
    fig.tight_layout()
    
    curves_path = "models/training_curves.png"
    plt.savefig(curves_path)
    plt.close()
    
    print(f"Final models and training curves plot saved to models/ directory (Curves: {curves_path}).")

if __name__ == "__main__":
    run_regression_pipeline()