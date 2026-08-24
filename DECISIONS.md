# DECISIONS.md

A log of every non-obvious design choice in this project.

---

### 001 — Plain scripts vs installable package
*   **Choice:** Flat Python scripts in `src/`, no package structure.
*   **Reason:** Simplifies development, Docker file copying, and GitOps tracking.

### 002 — yfinance for data ingestion
*   **Choice:** Ingest hourly candles from Yahoo Finance (`yfinance`).
*   **Reason:** Provides reliable, clean, free data feeds for `^NSEI` (Nifty 50) and `^INDIAVIX` with zero API authentication requirements, minimizing production API dependencies.

### 003 — Realized Volatility Regression vs Binary Classification
*   **Choice:** Model future 5-hour realized volatility as a continuous regression variable.
*   **Reason:** Volatility is highly clusters-based and mean-reverting, making it statistically easier to model and highly applicable for options straddle trading (pricing premiums).

### 004 — Target Scaling
*   **Choice:** Multiply raw realized volatility targets by 100.
*   **Reason:** Standard volatility decimals (e.g. `0.0008`) result in vanishing PyTorch gradients (MSE Loss is too close to zero to trigger weight updates). Scaling shifts the decimals to integers, ensuring smooth gradient descent.

### 005 — PyTorch Attention-GRU vs Classical ML (XGBoost)
*   **Choice:** Use a 2-layer Recurrent Neural Network (GRU) coupled with a Temporal Prior Attention Head.
*   **Reason:** Time-series volatility has long-term memory signatures that recurrent cells capture far better than tree-based models like XGBoost. The temporal attention head helps the network focus on specific market sessions (like market opens).

### 006 — Postgres + Redis Dual Database Architecture
*   **Choice:** Postgres (Supabase) for permanent training databases and drift logs; Redis (Upstash) for API cache.
*   **Reason:** Supabase manages relational performance logs, while Upstash Redis ensures sub-20ms response times for the prediction endpoint under high query volumes.

### 007 — Fractional Differentiation ($d=0.40$)
*   **Choice:** Apply fractional differentiation with order $d=0.40$ on Nifty prices.
*   **Reason:** Standard integer differentiation ($d=1$) removes all price history (memory). Fractional differentiation removes the non-stationary trend while keeping maximum historical price correlation, providing a cleaner feed to the neural network.


