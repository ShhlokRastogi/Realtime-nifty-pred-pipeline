#!/bin/bash
# 1. Add the src folder to Python's import search path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 2. Start the poller loop in the background
python src/poll.py &

# 3. Start the FastAPI API server in the foreground
uvicorn src.api:app --host 0.0.0.0 --port $PORT
