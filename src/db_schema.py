import psycopg2
from config import DB_CONFIG

def initialize_database_schema():
    """Creates raw data, features, training matrix, forecasts, and drift tables in Supabase."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # 1. Table for storing raw ingested Nifty 50 and India VIX hourly candles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nifty_vix_raw (
            datetime TIMESTAMP PRIMARY KEY,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume BIGINT,
            vix NUMERIC
        );
    """)
    
    # 2. Table for storing the merged 16-feature training matrix
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nifty_training_data (
            datetime TIMESTAMP PRIMARY KEY,
            close NUMERIC,
            rsi NUMERIC,
            macd_diff_pct NUMERIC,
            bb_width NUMERIC,
            atr_pct NUMERIC,
            hl_spread NUMERIC,
            volume_delta NUMERIC,
            lagged_return_1 NUMERIC,
            vix NUMERIC,
            vix_return NUMERIC,
            realized_vol_5 NUMERIC,
            realized_vol_10 NUMERIC,
            realized_vol_20 NUMERIC,
            close_fracdiff NUMERIC,
            sin_hour NUMERIC,
            cos_hour NUMERIC,
            sin_day NUMERIC,
            cos_day NUMERIC
        );
    """)
    
    # 3. Table for storing live model forecasts
    cur.execute("""
        CREATE TABLE IF NOT EXISTS volatility_forecasts (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker VARCHAR(20) DEFAULT '^NSEI',
            current_price NUMERIC,
            current_vix NUMERIC,
            current_realized_vol NUMERIC,
            forecasted_vol_5h NUMERIC,
            expected_change_pct NUMERIC,
            action VARCHAR(100),
            source_datetime TIMESTAMP,
            target_datetime TIMESTAMP,
            model_version VARCHAR(50) DEFAULT 'attention_gru_v1',
            CONSTRAINT uq_forecasts UNIQUE (ticker, source_datetime, model_version)
        );
    """)
    
    # Run column migrations to ensure existing deployments get the new columns
    cur.execute("""
        ALTER TABLE volatility_forecasts 
        ADD COLUMN IF NOT EXISTS source_datetime TIMESTAMP,
        ADD COLUMN IF NOT EXISTS target_datetime TIMESTAMP,
        ADD COLUMN IF NOT EXISTS model_version VARCHAR(50) DEFAULT 'attention_gru_v1';
    """)
    
    # Add unique constraint uq_forecasts if it does not exist
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_forecasts') THEN
                ALTER TABLE volatility_forecasts ADD CONSTRAINT uq_forecasts UNIQUE (ticker, source_datetime, model_version);
            END IF;
        END;
        $$;
    """)
    
    # 3b. Table for Postgres-backed distributed lock (Concurrency safety)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lock_store (
            lock_key VARCHAR(50) PRIMARY KEY,
            locked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 4. Table for tracking model performance & drift history (Full Regression Metrics)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_drift_metrics (
            id SERIAL PRIMARY KEY,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            evaluation_window_hours INT,
            mean_absolute_error NUMERIC,
            r2_score NUMERIC,
            directional_accuracy NUMERIC,
            accuracy_threshold NUMERIC,
            drift_detected BOOLEAN
        );
    """)
    
    # Create indexes to speed up query performance
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_forecasts_source_datetime ON volatility_forecasts (source_datetime DESC);
        CREATE INDEX IF NOT EXISTS idx_forecasts_target_datetime ON volatility_forecasts (target_datetime DESC);
        CREATE INDEX IF NOT EXISTS idx_drift_calculated_at ON model_drift_metrics (calculated_at DESC);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database tables initialized successfully in Supabase.")

if __name__ == "__main__":
    initialize_database_schema()