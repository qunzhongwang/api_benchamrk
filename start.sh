#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# --- Config (edit these) ---
export POE_API_KEY="${POE_API_KEY:-your_key_here}"
CONCURRENCY=20
NUM_QUESTIONS=100
MAX_TOKENS=256
MODEL="gpt-5.3-codex"
BASE_URL="https://api.poe.com/v1"
# ---------------------------

# Setup venv if needed
if [ ! -d ".venv" ]; then
    echo ">>> Creating venv with uv..."
    uv sync
fi

source .venv/bin/activate

echo ">>> Running stress test: ${NUM_QUESTIONS} questions, ${CONCURRENCY} concurrent"
python main.py \
    --api-key "$POE_API_KEY" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --num-questions "$NUM_QUESTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --show-all
