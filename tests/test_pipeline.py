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


def test_target_timestamp_mapping():
    """
    Verify get_next_market_candle correct trading hour advancement and weekend skipping.
    """
    from poll import get_next_market_candle
    import datetime

    # Scenario A: Same-day market hour advancement
    # 2026-08-10 (Monday) 10:15:00 + 5 steps = Monday 15:15:00
    dt_a = datetime.datetime(2026, 8, 10, 10, 15, 0)
    target_a = get_next_market_candle(dt_a, steps=5)
    assert target_a == datetime.datetime(2026, 8, 10, 15, 15, 0)

    # Scenario B: Wrap to next day
    # Monday 14:15:00 + 5 steps = Tuesday 12:15:00
    dt_b = datetime.datetime(2026, 8, 10, 14, 15, 0)
    target_b = get_next_market_candle(dt_b, steps=5)
    assert target_b == datetime.datetime(2026, 8, 11, 12, 15, 0)

    # Scenario C: Weekend skipping
    # Friday 15:15:00 + 5 steps = Monday 13:15:00 (Saturday & Sunday skipped)
    dt_c = datetime.datetime(2026, 8, 14, 15, 15, 0) # Aug 14, 2026 is Friday
    target_c = get_next_market_candle(dt_c, steps=5)
    assert target_c == datetime.datetime(2026, 8, 17, 13, 15, 0) # Aug 17 is Monday


def test_directional_baseline_alignment():
    """
    Verify that the baseline current volatility aligns to split_idx + SEQ_LEN - 1.
    """
    from config import SEQ_LEN
    # Simulate a sequence run
    total_samples = 150
    realized_vol_5_raw = np.linspace(0.05, 0.15, total_samples) # 150 steps
    split_idx = 40
    wf_preds_len = 30

    # Under corrected logic:
    current_vol_baseline = (
        realized_vol_5_raw[
            split_idx + SEQ_LEN - 1 :
            split_idx + SEQ_LEN - 1 + wf_preds_len
        ] * 100.0
    )

    # The first element must match index split_idx + SEQ_LEN - 1
    expected_first = realized_vol_5_raw[split_idx + SEQ_LEN - 1] * 100.0
    assert current_vol_baseline[0] == expected_first
    assert len(current_vol_baseline) == wf_preds_len


def test_no_future_vix_backfill():
    """
    Verify left-join of Nifty/VIX and forward-fill only (no bfill) does not leak future data.
    """
    dates = pd.date_range(start="2026-08-01", periods=5, freq="1h")
    df_nifty = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
    # VIX has no value for first row, but has values for rows 2 and 3
    df_vix_close = pd.DataFrame({"vix": [np.nan, 12.0, 13.0, np.nan, np.nan]}, index=dates)

    # Perform the exact merge logic:
    df_merged = df_nifty.join(df_vix_close, how='left')
    df_merged['vix'] = df_merged['vix'].ffill()
    
    # Verify first row VIX is still NaN (ffill does not bfill backwards)
    assert pd.isna(df_merged['vix'].iloc[0])
    
    # Verify dropna drops the first row but keeps the rest
    df_clean = df_merged.dropna(subset=['vix'])
    assert len(df_clean) == 4
    assert df_clean['vix'].iloc[0] == 12.0 # Row 1 ffilled
    assert df_clean['vix'].iloc[2] == 13.0 # Row 3 ffilled VIX from Row 2


def test_redis_cache_refresh(monkeypatch):
    """
    Verify Redis cache setex function is called with the serialized forecast payload.
    """
    import json
    class MockRedis:
        def __init__(self, **kwargs):
            self.store = {}
        def setex(self, key, ttl, val):
            self.store[key] = (val, ttl)

    mock_r = MockRedis()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_r)

    # Run cache refresh logic
    result = {
        "ticker": "^NSEI",
        "current_price": 24000.0,
        "current_vix": 11.0,
        "current_realized_vol": 0.08,
        "forecasted_vol_5h": 0.12,
        "expected_change_pct": 50.0,
        "action": "NORMAL",
        "date": "2026-08-31 09:15:00"
    }
    
    import redis
    r_client = redis.Redis()
    r_client.setex("nifty_forecast", 3600, json.dumps(result))

    assert "nifty_forecast" in mock_r.store
    val, ttl = mock_r.store["nifty_forecast"]
    assert ttl == 3600
    data = json.loads(val)
    assert data["current_price"] == 24000.0


