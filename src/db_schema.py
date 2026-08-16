"""
Database schema setup for the crypto prediction pipeline.
Creates the ohlcv and features tables in Postgres.

Run once: python src/db_schema.py
"""
import psycopg2
from config import DB_CONFIG


def create_tables():
    """Create the ohlcv and features tables if they don't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Table 1: Raw OHLCV price data
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20) NOT NULL,
            date        TIMESTAMP NOT NULL,
            open        DOUBLE PRECISION,
            high        DOUBLE PRECISION,
            low         DOUBLE PRECISION,
            close       DOUBLE PRECISION,
            volume      DOUBLE PRECISION,
            UNIQUE(ticker, date)  -- prevent duplicate rows for same ticker+date
        );
    """)

    # Table 2: Computed features (derived from ohlcv)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id                  SERIAL PRIMARY KEY,
            ticker              VARCHAR(20) NOT NULL,
            date                TIMESTAMP NOT NULL,
            rsi                 DOUBLE PRECISION,
            sma_crossover       DOUBLE PRECISION,
            rolling_volatility  DOUBLE PRECISION,
            volume_delta        DOUBLE PRECISION,
            lagged_return_1     DOUBLE PRECISION,
            lagged_return_3     DOUBLE PRECISION,
            lagged_return_5     DOUBLE PRECISION,
            UNIQUE(ticker, date)  -- one feature row per ticker per day
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully.")


if __name__ == "__main__":
    create_tables()
