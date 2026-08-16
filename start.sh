#!/bin/bash
# 1. Start the poller loop in the background
python src/poll.py &

# 2. Start the FastAPI API server in the foreground
uvicorn src.api:app --host 0.0.0.0 --port $PORT
