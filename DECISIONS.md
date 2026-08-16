# DECISIONS.md

A log of every non-obvious design choice in this project.

---

### 001 — Plain scripts vs installable package
- **Choice:** Flat Python scripts in `src/`, no `setup.py` or `pyproject.toml`.
- **Alternative:** Proper Python package with `pyproject.toml` and `pip install -e .`
- **Reason:** We don't need installability yet; flat scripts are easier to navigate and debug while learning. We'll add packaging when we build the FastAPI serving layer.

### 002 — yfinance vs Binance API for historical data
- **Choice:** yfinance for Week 1 (daily candles).
- **Alternative:** Binance public REST API (supports 1m/1h candles, more reliable).
- **Reason:** yfinance is a one-liner with no auth; sufficient for daily data. Binance adds pagination and rate-limit handling that isn't needed until we go to finer granularity in Week 2.

### 003 — Binary (up/down) vs 3-class (up/down/flat) for baseline
- **Choice:** Binary classification (sign of next-day return) for Week 1.
- **Alternative:** 3-class with a flat threshold, or the two-stage cascaded classifier (flat-detector → direction-classifier).
- **Reason:** Binary gives cleaner class balance and a simpler first model. The two-stage design is planned for Week 2–3 once we have return distributions to pick a good flat threshold.

### 004 — SMA vs EMA for crossover feature
- **Choice:** SMA(10) vs SMA(50) crossover.
- **Alternative:** EMA, which weights recent prices more heavily.
- **Reason:** At daily granularity the difference is marginal; SMA is easier to inspect and debug when sanity-checking features.

### 005 — XGBoost vs logistic regression for baseline
- **Choice:** XGBoost as the primary model, with a dummy (majority-class) classifier as the floor.
- **Alternative:** Logistic regression as an intermediate baseline.
- **Reason:** XGBoost is in the project spec and is still lightweight enough for fast iteration. The dummy classifier gives us the "does this model beat random?" check that logistic regression would have served.

### 006 — Postgres + Redis Dual Database Architecture
- **Choice:** Postgres for permanent historical tables (offline store) + Redis for latest values (online cache).
- **Alternative:** Postgres only.
- **Reason:** Postgres handles robust SQL querying and multi-process updates without file corruption. Redis ensures prediction endpoint latency stays under <1ms.

### 007 — Binance API for Live Polling
- **Choice:** Fetching live data via Binance API in `poll.py`.
- **Alternative:** yfinance.
- **Reason:** Binance's public REST endpoints provide real-time updates and spot prices instantly, whereas yfinance is slow and unsuitable for live-updating pipelines.

