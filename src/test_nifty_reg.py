import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, VOL_FORECAST_WINDOW, SEQ_LEN, LOOKBACK_SIZE
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definitions matching training
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

def run_volatility_backtest():
    # Load processed data
    processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_features.csv")
    df = pd.read_csv(processed_path, index_col='Datetime', parse_dates=True)
    
    FEATURE_COLS_VOL = [
        "rsi", "macd_diff_pct", "bb_width", "atr_pct", "hl_spread", "volume_delta", 
        "lagged_return_1", "vix", "vix_return", "realized_vol_5", "realized_vol_10", 
        "realized_vol_20", "sin_hour", "cos_hour", "sin_day", "cos_day"
    ]
    
    X_raw_continuous = df[FEATURE_COLS_VOL].values
    closes = df['close'].values
    realized_vol_5_raw = df['realized_vol_5'].values
    
    # Load model and scaler
    model_path = "models/attention_regressor.pt"
    scaler_path = "models/scaler_regressor.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model files missing. Run train_reg.py first.")
        
    model = AttentionGRURegressor(input_dim=16, hidden_dim=256, num_layers=3, dropout=0.19345).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(scaler_path, "rb") as f_in:
        scaler = pickle.load(f_in)
        
    X_seq_list, y_seq_list = [], []
    for i in range(len(df) - SEQ_LEN - VOL_FORECAST_WINDOW):
        seq_x = X_raw_continuous[i : i + SEQ_LEN]
        future_vols = realized_vol_5_raw[i + SEQ_LEN - 1 + 1 : i + SEQ_LEN - 1 + 1 + VOL_FORECAST_WINDOW]
        avg_future_vol = np.mean(future_vols)
        X_seq_list.append(seq_x)
        y_seq_list.append(avg_future_vol * 100.0) # Scaled target
        
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)
    
    split_idx = int(len(X_seq) * 0.8)
    
    # Isolate test slice
    X_test = X_seq[split_idx:]
    y_test_actuals = y_seq[split_idx:]
    
    # Predict out-of-sample
    N_ts, S, F = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(N_ts, S, F)
    
    print("Generating out-of-sample volatility forecasts...")
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        preds = model(X_test_tensor).cpu().numpy()
        
    current_vol_baseline = (
        realized_vol_5_raw[
            split_idx + SEQ_LEN - 1:
            split_idx + SEQ_LEN - 1 + len(preds)
        ] * 100.0
    )
    
    # =====================================================================
    # SIMULATE OPTIONS VOLATILITY STRATEGY (Fixed 2% Risk Allocation)
    # =====================================================================
    print("Simulating Options Straddle Volatility Strategy (2% Risk allocation)...")
    initial_capital = 100000.0  # ₹1,00,000 starting capital
    capital = initial_capital
    portfolio_history = [capital]
    
    theta_decay_penalty = 0.05 / 100.0  # Theta decay per hour
    risk_allocation = 0.02  # Risk exactly 2% of our total portfolio capital per trade
    
    wins = 0
    losses = 0
    trades_taken = 0
    
    for t in range(len(preds)):
        pred_vol = preds[t]
        curr_vol = current_vol_baseline[t]
        actual_vol = y_test_actuals[t]
        
        # Calculate expected volatility change
        expected_change = (pred_vol - curr_vol) / (curr_vol + 1e-9)
        
        # Trade if model expects >10% move in volatility
        if expected_change > 0.10:
            signal = 1  # Long Volatility (Buy Straddle)
        elif expected_change < -0.10:
            signal = -1  # Short Volatility (Sell Straddle)
        else:
            signal = 0  # Hold cash
            
        if signal != 0:
            trades_taken += 1
            actual_vol_change = (actual_vol - curr_vol) / (curr_vol + 1e-9)
            
            # Trade return on the options contract itself
            trade_return = (signal * actual_vol_change) - theta_decay_penalty
            
            # Risk Management: Cap maximum loss at -100% (total premium loss)
            trade_return = max(trade_return, -1.0)
            
            # Realized profit/loss on the portfolio (only 2% of capital was risked)
            portfolio_change = trade_return * risk_allocation
            
            capital = capital * (1 + portfolio_change)
            
            if trade_return > 0:
                wins += 1
            else:
                losses += 1
                
        portfolio_history.append(capital)
        
    portfolio_history = np.array(portfolio_history)
    final_return = ((capital - initial_capital) / initial_capital) * 100
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    
    print("\n" + "="*20 + " REALISTIC VOLATILITY BACKTEST RESULTS " + "="*20)
    print(f"Starting Capital       : ₹{initial_capital:,.2f}")
    print(f"Final Portfolio Value  : ₹{capital:,.2f}")
    print(f"Total Return           : {final_return:.2f}%")
    print(f"Total Trades Executed  : {trades_taken}")
    print(f"Options Win Rate       : {win_rate:.2f}%")
    print("="*60)
    
    # Save Equity Curve Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history, label="Options Volatility Bot (Dynamic Straddle)", color="green", linewidth=2.5)
    plt.axhline(y=initial_capital, color="red", linestyle="--", alpha=0.5, label="Starting Capital")
    plt.title("Nifty 50 Options Straddle Strategy (Compounding Growth - 2% Risk)")
    plt.xlabel("Hours of Trading")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "models/volatility_backtest.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Equity Curve graph saved to: {plot_path}")

if __name__ == "__main__":
    run_volatility_backtest()