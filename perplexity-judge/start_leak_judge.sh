#!/bin/bash
# Start Perplexity Leak Judge Service
#
# Usage:
#   ./start_leak_judge.sh          # Default: GPU 7, port 8765
#   ./start_leak_judge.sh 6 8766   # Custom: GPU 6, port 8766

set -e

GPU_ID=${1:-7}
PORT=${2:-8765}
MODEL_PATH=${LEAK_JUDGE_MODEL_PATH:-"Qwen/Qwen3-0.6B"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Perplexity Leak Judge Service"
echo "=============================================="
echo "GPU:        $GPU_ID (appears as cuda:0 to process)"
echo "Port:       $PORT"
echo "Model:      $MODEL_PATH"
echo "Endpoint:   http://localhost:$PORT"
echo "=============================================="

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port $PORT is already in use"
    echo "Kill existing process or use a different port"
    exit 1
fi

# Start server
cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID \
LEAK_JUDGE_MODEL_PATH="$MODEL_PATH" \
LEAK_JUDGE_DEVICE="cuda:0" \
python server.py --port $PORT

