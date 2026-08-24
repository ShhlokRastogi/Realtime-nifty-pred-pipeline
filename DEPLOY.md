# Deployment Guide: Free Cloud Hosting ($0/Month)

This document details the step-by-step instructions to deploy your Git-Ops prediction pipeline to the cloud for free using a hybrid-cloud architecture.

---

## The Architecture Overview

```
   [ Render Web Service ] <───(5m ping keeps awake)─── [ UptimeRobot ]
   (Runs FastAPI + poll.py)
              │
      ┌───────┴───────┐
      ▼               ▼
[ Supabase ]     [ Upstash ]
(Postgres DB)   (Redis Cache)
```

---

## Phase 1: Set Up Cloud Databases

### 1. Supabase (Free PostgreSQL)
1. Go to [Supabase.com](https://supabase.com) and create a free account.
2. Create a new project (e.g. `crypto-db`). Set a database password and save it securely.
3. Once the database is ready, go to **Project Settings** -> **Database**.
4. Scroll down to **Connection String**, select **URI**, and copy the string. It will look like:
   `postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres`

### 2. Upstash (Free Serverless Redis)
1. Go to [Upstash.com](https://upstash.com) and create a free account.
2. Create a new Redis database.
3. Under **Connection Details**, copy the **Endpoint**, **Port**, and **Password**.

---

## Phase 2: Code Configurations

To deploy, we must update our local config to read from cloud environment variables. 

### 1. Update [`src/config.py`](file:///C:/D/crypto/src/config.py)
Modify the database and Redis configs to read from environment variables:
```python
import os

# PostgreSQL Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
    "database": os.getenv("DB_NAME", "crypto_features"),
}

# Redis Configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "password": os.getenv("REDIS_PASSWORD", None),
}
```

### 2. Create the Startup Script (`start.sh`)
Since Render only allows one free web service, we will bundle the poller (`poll.py`) and FastAPI (`api.py`) inside the same container. 

Create a file named `start.sh` in your project root:
```bash
#!/bin/bash
# 1. Start the poller loop in the background
python src/poll.py &

# 2. Start the FastAPI API server in the foreground
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

---

## Phase 3: Deploy to Render

1. Go to [Render.com](https://render.com) and sign up.
2. Click **New** -> **Web Service**.
3. Link your GitHub repository `self-healing-crypto-pipeline`.
4. Set the configurations:
   * **Runtime:** `Python`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `bash start.sh`
   * **Instance Type:** `Free`
5. Click **Advanced** and add these **Environment Variables**:
   * `DB_HOST`: *(Your Supabase Host)*
   * `DB_USER`: `postgres`
   * `DB_PASSWORD`: *(Your Supabase Password)*
   * `DB_NAME`: `postgres`
   * `DB_PORT`: `5432`
   * `REDIS_HOST`: *(Your Upstash Endpoint)*
   * `REDIS_PORT`: *(Your Upstash Port)*
   * `REDIS_PASSWORD`: *(Your Upstash Password)*
6. Click **Create Web Service**.

---

## Phase 4: Initialize the Cloud Database Schema

Before the app starts, Postgres needs the table schemas. Run this command **locally on your computer** to connect to Supabase once and create the tables:

```powershell
# Set database variables temporarily to point to Supabase
$env:DB_HOST="[YOUR-SUPABASE-HOST]"
$env:DB_PASSWORD="[YOUR-SUPABASE-PASSWORD]"
$env:DB_USER="postgres"
$env:DB_NAME="postgres"

# Run schema creator
python src/db_schema.py

# Run ingestion once to load the 3-year baseline to the cloud
python src/ingest.py
python src/feature_store.py
```

---

## Phase 5: Keep the Server Awake

Render's free tier goes to sleep after 15 minutes of inactivity. To prevent this:
1. Go to [UptimeRobot.com](https://uptimerobot.com) and create a free account.
2. Click **Add New Monitor**.
3. Set:
   * **Monitor Type:** `HTTP(s)`
   * **Friendly Name:** `Crypto API`
   * **URL:** `https://[YOUR-RENDER-SUBDOMAIN].onrender.com/health`
   * **Monitoring Interval:** `Every 5 minutes`
4. Save the monitor. UptimeRobot will ping your endpoint every 5 minutes, keeping the server (and your poller) awake 24/7!
