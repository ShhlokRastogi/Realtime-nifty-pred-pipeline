import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
from config import DB_CONFIG, TEST_CUTOFF

FEATURE_COLS = [
    "rsi", "sma_crossover", "rolling_volatility",
    "volume_delta", "lagged_return_1", "lagged_return_3", "lagged_return_5"
]

def load_baseline_features(ticker: str, limit: int = 2000) -> pd.DataFrame:
    """
    Load baseline features for a given ticker from the Postgres database.
    Limits to the latest N rows before the cutoff to prevent memory exhaustion (OOM).

    Parameters:
    ticker (str): The ticker symbol to load features for.
    limit (int): Maximum number of baseline rows to load.

    Returns:
    pd.DataFrame: DataFrame containing the features, indexed by date.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"""
        SELECT date, {', '.join(FEATURE_COLS)}
        FROM features
        WHERE ticker = %s and date < %s
        ORDER BY date DESC
        LIMIT %s;
    """
    df = pd.read_sql(query, conn, params=(ticker, TEST_CUTOFF, limit))
    conn.close()
    df.set_index('date', inplace=True)
    df = df.sort_index()
    return df

def load_recent_features(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Loads the most recent N feature records from Postgres."""
    conn = psycopg2.connect(**DB_CONFIG)
    # We order DESCENDING to get the latest rows, and limit the count
    query = f"""
        SELECT date, {', '.join(FEATURE_COLS)}
        FROM features
        WHERE ticker = %s and date >= %s
        ORDER BY date DESC
        LIMIT %s;
    """
    df = pd.read_sql(query, conn, params=(ticker, TEST_CUTOFF, limit))
    conn.close()
    
    # Set the index
    df.set_index('date', inplace=True)
    
    # Sort the index back ascending so the time-series matches the baseline order
    df = df.sort_index()
    return df

def calculate_feature_pvalues(baseline_df: pd.DataFrame, recent_df: pd.DataFrame) -> dict:
    """
    Runs the Kolmogorov-Smirnov test for each feature.
    
    Returns:
        dict mapping feature_name -> p-value (float)
    """
    p_values = {}
    for col in FEATURE_COLS:
        # 1. Check if the column exists in both dataframes
        # 2. Get the series, drop NaNs: .dropna()
        # 3. If both have data, run: _, p_val = ks_2samp(base_vals, recent_vals)
        # 4. Store: p_values[col] = float(p_val)
        if col in baseline_df.columns and col in recent_df.columns:
            base_vals = baseline_df[col].dropna()
            recent_vals = recent_df[col].dropna()
            if not base_vals.empty and not recent_vals.empty:
                _, p_val = ks_2samp(base_vals, recent_vals)
                p_values[col] = float(p_val)
        
    return p_values
def is_drift_detected(p_values: dict) -> bool:
    """
    Consensus Drift Logic.
    Trigger drift if 2 or more features show significant shift (p-value < 0.05).
    """
    drift_count = 0
    
    # 1. Loop through all the p-values in the dictionary
    # 2. If a p-value is less than 0.05, increment drift_count
    # 3. If drift_count is greater than or equal to 2, return True
    # 4. Otherwise, return False at the end
    
    # Write your code here:
    for p_val in p_values.values():
        if p_val < 0.05:
            drift_count += 1

    return drift_count >= 2


def run_drift_check(ticker: str) -> dict:
    """Runs the full drift check cycle for a single ticker."""
    baseline = load_baseline_features(ticker)
    recent = load_recent_features(ticker, limit=100)
    
    if len(baseline) < 20 or len(recent) < 20:
        return {
            "status": "error", 
            "message": f"Not enough data to compute drift. Baseline: {len(baseline)} rows, Recent: {len(recent)} rows"
        }
        
    p_values = calculate_feature_pvalues(baseline, recent)
    drift_flag = is_drift_detected(p_values)
    
    return {
        "status": "success",
        "ticker": ticker,
        "p_values": p_values,
        "drift_detected": drift_flag
    }


if __name__ == "__main__":
    # Test run for BTC-USD
    import json
    res = run_drift_check("BTC-USD")
    print(json.dumps(res, indent=2))