"""
Feature Store — computes features from raw OHLCV data in Postgres,
writes them back to the features table, and caches the latest row in Redis.

Run: python src/feature_store.py
"""
import json
import pandas as pd
import psycopg2
import redis
from features import build_features
from config import DB_CONFIG, REDIS_CONFIG, TICKERS


def read_ohlcv_from_db(ticker: str, limit: int = None) -> pd.DataFrame:
    """
    Read raw OHLCV data for a ticker from the Postgres ohlcv table.
    Supports an optional limit to read only the latest N rows.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    conn = psycopg2.connect(**DB_CONFIG)
    if limit:
        query = """
            SELECT date, open, high, low, close, volume
            FROM ohlcv
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(ticker, limit), parse_dates=["date"])
        # Reverse the order to ASC so rolling windows calculate chronologically
        df = df.iloc[::-1]
    else:
        query = """
            SELECT date, open, high, low, close, volume
            FROM ohlcv
            WHERE ticker = %s
            ORDER BY date ASC
        """
        df = pd.read_sql(query, conn, params=(ticker,), parse_dates=["date"])
        
    conn.close()

    # Set date as index (same format as the CSV-based pipeline)
    df = df.set_index("date")
    return df


from psycopg2.extras import execute_values


def save_features_to_db(df: pd.DataFrame, ticker: str) -> int:
    """
    Write computed features to the Postgres features table using fast bulk inserts.
    Uses ON CONFLICT to upsert (update if row already exists).
    
    Returns:
        Number of rows written.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Convert DataFrame rows into a list of tuples
    tuples = [
        (
            ticker,
            date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
            float(row["rsi"]) if not pd.isna(row["rsi"]) else None,
            float(row["sma_crossover"]) if not pd.isna(row["sma_crossover"]) else None,
            float(row["rolling_volatility"]) if not pd.isna(row["rolling_volatility"]) else None,
            float(row["volume_delta"]) if not pd.isna(row["volume_delta"]) else None,
            float(row["lagged_return_1"]) if not pd.isna(row["lagged_return_1"]) else None,
            float(row["lagged_return_3"]) if not pd.isna(row["lagged_return_3"]) else None,
            float(row["lagged_return_5"]) if not pd.isna(row["lagged_return_5"]) else None,
        )
        for date, row in df.iterrows()
    ]

    query = """
        INSERT INTO features (ticker, date, rsi, sma_crossover, rolling_volatility,
                              volume_delta, lagged_return_1, lagged_return_3, lagged_return_5)
        VALUES %s
        ON CONFLICT (ticker, date) DO UPDATE
        SET rsi = EXCLUDED.rsi,
            sma_crossover = EXCLUDED.sma_crossover,
            rolling_volatility = EXCLUDED.rolling_volatility,
            volume_delta = EXCLUDED.volume_delta,
            lagged_return_1 = EXCLUDED.lagged_return_1,
            lagged_return_3 = EXCLUDED.lagged_return_3,
            lagged_return_5 = EXCLUDED.lagged_return_5;
    """

    # execute_values runs a highly optimized bulk INSERT
    execute_values(cur, query, tuples)

    conn.commit()
    cur.close()
    conn.close()
    return len(tuples)



def cache_latest_in_redis(df: pd.DataFrame, ticker: str) -> None:
    """
    Cache the most recent feature row for a ticker in Redis.
    Stored as a JSON string under key "features:{ticker}".

    The FastAPI endpoint reads this for instant predictions.
    """
    r = redis.Redis(**REDIS_CONFIG)

    # Get the last row (most recent date)
    latest = df.iloc[-1]
    feature_cols = [
        "rsi", "sma_crossover", "rolling_volatility",
        "volume_delta", "lagged_return_1", "lagged_return_3", "lagged_return_5"
    ]

    payload = {
        "ticker": ticker,
        "date": str(latest.name),
        "features": {col: float(latest[col]) for col in feature_cols}
    }

    # Store as JSON string with key "features:BTC-USD" etc.
    r.set(f"features:{ticker}", json.dumps(payload))
    print(f"  Cached latest features in Redis: features:{ticker}")


def run_feature_store(limit: int = None):
    """
    Full feature store pipeline:
    1. Read OHLCV from Postgres
    2. Compute features
    3. Write features to Postgres
    4. Cache latest row in Redis
    """
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")

        # Step 1: Read raw data from Postgres
        ohlcv_df = read_ohlcv_from_db(ticker, limit=limit)
        print(f"  Read {len(ohlcv_df)} OHLCV rows from Postgres")

        # Step 2: Compute features
        features_df = build_features(ohlcv_df.copy())
        print(f"  Computed {len(features_df)} feature rows (dropped {len(ohlcv_df) - len(features_df)} NaN rows from rolling windows)")

        # Step 3: Write features to Postgres
        count = save_features_to_db(features_df, ticker)
        print(f"  Saved {count} feature rows to Postgres")

        # Step 4: Cache latest in Redis
        cache_latest_in_redis(features_df, ticker)


if __name__ == "__main__":
    run_feature_store()
