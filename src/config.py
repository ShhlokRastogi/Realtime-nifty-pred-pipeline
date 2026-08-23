"""
Shared configuration constants for the market volatility prediction pipeline.
"""

# ── Tickers ──────────────────────────────────────────────
# ^NSEI: Nifty 50 Index (Target)
# ^INDIAVIX: India VIX (Expectation Index)
TICKERS = ["^NSEI", "^INDIAVIX"]

# ── Date range for historical pull ───────────────────────
HIST_START = "2024-08-22"
HIST_END = "2026-08-22"

# ── Feature parameters ───────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
ATR_WINDOW = 14
FFD_D = 0.40
FFD_MAX_LAGS = 60

# ── Volatility Forecasting ──────────────────────────────
VOL_FORECAST_WINDOW = 5  # Predict next 5 hours of volatility
LOOKBACK_SIZE = 1200     # Training history per rolling step
SEQ_LEN = 42             # Sequence length (7 trading days)

# ── Data paths ───────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# ── Model ────────────────────────────────────────────────
RANDOM_SEED = 42

import os

# ── Database (Postgres / Supabase) ─────────────────────
# Reads from Render/Supabase cloud variables in production
# Falls back to local docker-compose settings in development
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
    "dbname": os.getenv("DB_NAME", "crypto_features"),
}

# ── Cache (Redis / Upstash) ────────────────────────────
# Reads from Upstash serverless variables in production
# Falls back to local redis service in development
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "db": 0,
}