#!/bin/bash
# Generate SFT training data for the Refiner by calling DeepSeek-R1 to rephrase
# math problems. Reads source questions and writes (prompt, response) parquet/jsonl
# ready for `run_sft_qwen3_4b.sh`.
#
# Set DEEPSEEK_API_KEY before running.

set -eu

: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY for rephrase generation}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$(dirname "$0")"

echo "=================================="
echo "Generating Refiner SFT data"
echo "=================================="
echo "Source : raw math questions (GSM8K + MATH mix)"
echo "Target : rephrase pairs via DeepSeek-R1"
echo "Output : mixed_sft_with_responses.{parquet,jsonl}"
echo "=================================="

python generate_sft_data.py

echo
echo "Done. Inspect the jsonl, then train with run_sft_qwen3_4b.sh."
