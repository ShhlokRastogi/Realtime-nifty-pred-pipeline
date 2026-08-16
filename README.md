# Crypto Price-Direction Prediction Pipeline

A self-healing ML pipeline that predicts short-term BTC/ETH price direction.

## Current Status: Week 1 — Static Baseline

Building a baseline XGBoost model on daily OHLCV data from yfinance.

## Project Structure

```
crypto/
├── data/              # Raw + processed data (gitignored)
├── src/
│   ├── config.py      # Shared constants (tickers, feature params, paths)
│   ├── ingest.py      # Data ingestion from yfinance
│   ├── features.py    # Feature engineering (RSI, SMA, volatility, etc.)
│   └── train.py       # Model training, evaluation, MLflow logging
├── Notebooks/         # Exploration and analysis
├── requirements.txt   # Python dependencies
├── DECISIONS.md       # Log of every non-obvious design choice
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Roadmap

- **Week 1:** Static baseline (XGBoost on daily OHLCV features)
- **Week 2:** Live feature store (Postgres/Redis + finer granularity data)
- **Week 3:** Drift detection + Grafana dashboard
- **Week 4:** Auto-retrain + human approval gate
