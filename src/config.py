"""
Shared configuration constants for the crypto prediction pipeline.
"""

# ── Tickers ──────────────────────────────────────────────
from prometheus_client import Gauge


TICKERS = ["BTC-USD", "ETH-USD"]

# ── Date range for historical pull ───────────────────────
# ~2 years of daily data for training + evaluation
HIST_START = "2024-08-01"
HIST_END = "2026-08-11"  # yesterday

# ── Feature parameters ───────────────────────────────────
RSI_PERIOD = 14
SMA_SHORT = 10
SMA_LONG = 50
VOLATILITY_WINDOW = 20      # rolling std of log-returns
VOLUME_DELTA_WINDOW = 20    # rolling mean window for volume
LAGGED_RETURN_PERIODS = [1, 3, 5]  # days to look back

# ── Train/test split ────────────────────────────────────
# Time-series split: everything before this date = train, after = test
# Roughly last 3 months reserved for testing
TEST_CUTOFF = "2026-05-01"

# ── Data paths ───────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# ── Model ────────────────────────────────────────────────
RANDOM_SEED = 42

import os

# ── Database (Postgres) ─────────────────────────────────
# Reads environment variables for cloud deployment, falls back to local docker configs
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
    "dbname": os.getenv("DB_NAME", "crypto_features"),
}

# ── Cache (Redis) ────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_CONFIG = {
    "host": REDIS_HOST,
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "db": 0,
}

# Automatically enable SSL for cloud Redis (Upstash) connections
if "localhost" not in REDIS_HOST:
    REDIS_CONFIG["ssl"] = True
    REDIS_CONFIG["ssl_cert_reqs"] = "none"

DRIFT_GAUGE = Gauge(
    "drift_detected",                          # name
    "Drift detected for a given ticker",       # documentation
    ["ticker"],                                # labelnames
    namespace="crypto_mlops",
    subsystem="drift",
)
# TO QUERY IN PROMQL USE FORMAT: crypto_mlops_drift_drift_detected{ticker="BTC-USD"}
PVALUE_GAUGE = Gauge(
    "p_value",                                 # name
    "P-value for a given feature and ticker",  # documentation
    ["ticker", "feature"],                     # labelnames
    namespace="crypto_mlops",
    subsystem="drift",
)
# TO QUERY IN PROMQL USE FORMAT: crypto_mlops_drift_p_value{ticker="BTC-USD",feature="lagged_return_1"}
