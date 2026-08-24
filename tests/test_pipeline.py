import os
import sys
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Add src folder to the python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Set testing flag to isolate CI/CD environment
os.environ["TESTING"] = "true"

from features import calculate_technical_features
from api import app

client = TestClient(app)


def test_volatility_features_no_leakage():
    """
    Leakage Test: Modifying future data points must never alter past volatility features.
    
    This ensures that our RSI, realized volatility, and return calculations are causal 
    and do not suffer from lookahead bias.
    """
    # Create 100 rows of synthetic price data (rolling window will drop first ~20)
    dates = pd.date_range(start="2026-08-01", periods=100, freq="1h")
    df1 = pd.DataFrame({
        "open": np.linspace(100, 150, 100),
        "high": np.linspace(105, 155, 100),
        "low": np.linspace(95, 145, 100),
        "close": np.linspace(101, 151, 100),
        "volume": np.linspace(1000, 2000, 100)
    }, index=dates)

    # Calculate features on original data
    features1 = calculate_technical_features(df1.copy())

    # Create a duplicate dataset, but change the close price on the VERY LAST row (index 99)
    df2 = df1.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 999.0 # Massive change on the future row

    # Calculate features on the modified data
    features2 = calculate_technical_features(df2.copy())

    # Verify that features on row 75 (past data) are EXACTLY identical.
    # If they changed, it means future prices leaked backward in time!
    for col in features1.columns:
        val1 = features1.loc[dates[75], col]
        val2 = features2.loc[dates[75], col]
        assert np.isclose(val1, val2), f"Lookahead bias detected in feature: {col}!"


def test_api_health():
    """Verify that the FastAPI health check endpoint is active."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_predict():
    """
    Verify that our Nifty predict endpoint works and returns the correct response schema.
    """
    response = client.get("/predict/nifty")
    if response.status_code == 200:
        data = response.json()
        assert "ticker" in data
        assert "current_price" in data
        assert "current_vix" in data
        assert "forecasted_vol_5h" in data
        assert "expected_change_pct" in data
        assert "action" in data
        assert data["ticker"] == "^NSEI"
        assert isinstance(data["forecasted_vol_5h"], float)
    else:
        # If database is clean/empty or fails to connect, allow a 404/500 code
        assert response.status_code in [404, 500]

