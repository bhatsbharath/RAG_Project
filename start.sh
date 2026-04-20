#!/bin/bash
set -e

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
python main.py &
FASTAPI_PID=$!

# Wait until the health endpoint returns HTTP 200
echo "Waiting for API to be ready..."
until python -c "
import requests, sys
try:
    r = requests.get('http://localhost:8080/health', timeout=5)
    r.raise_for_status()
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    sleep 3
done
echo "API is ready. Starting Streamlit UI..."

# Start Streamlit UI (foreground — keeps the container alive)
streamlit run ui.py \
    --server.port=8000 \
    --server.address=0.0.0.0 \
    --server.headless=true
