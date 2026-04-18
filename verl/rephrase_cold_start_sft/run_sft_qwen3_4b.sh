#!/bin/bash
# Phase-1 cold-start SFT for the ReQueR Refiner.
#
# Trains Qwen3-4B on curated (raw_query, refined_query) pairs to establish
# the output format and baseline refining capability, then merges FSDP shards
# back to a single HuggingFace checkpoint.
#
# Set MODEL_PATH, TRAIN_DATA, CHECKPOINT_DIR, and MERGED_OUTPUT_DIR before running.

set -eu

: "${MODEL_PATH:?set MODEL_PATH to the base model (e.g. Qwen3-4B-thinking)}"
: "${TRAIN_DATA:?set TRAIN_DATA to the SFT parquet (see generate_sft_data.py)}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR for FSDP checkpoint shards}"
: "${MERGED_OUTPUT_DIR:?set MERGED_OUTPUT_DIR for the merged HF checkpoint}"

echo "==========================================="
echo "ReQueR Cold-Start SFT - Qwen3-4B"
echo "==========================================="
echo "Base model     : $MODEL_PATH"
echo "Train data     : $TRAIN_DATA"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "Merged output  : $MERGED_OUTPUT_DIR"
echo "==========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=3072 \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=32 \
    model.partial_pretrain="$MODEL_PATH" \
    model.fsdp_config.model_dtype=bfloat16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    optim.lr=5e-5 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    optim.clip_grad=1.0 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.project_name=requer-sft \
    trainer.experiment_name=qwen3-4b-sft \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger=console

echo "==========================================="
echo "SFT complete. Merging FSDP shards -> HuggingFace checkpoint"
echo "==========================================="

LATEST_ITER_FILE="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
if [ ! -f "$LATEST_ITER_FILE" ]; then
    echo "WARN: ${LATEST_ITER_FILE} not found; skipping merge." >&2
    exit 0
fi

LATEST_ITER=$(cat "$LATEST_ITER_FILE")
ACTOR_CHECKPOINT="${CHECKPOINT_DIR}/global_step_${LATEST_ITER}"

if [ ! -d "$ACTOR_CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at ${ACTOR_CHECKPOINT}" >&2
    exit 1
fi

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$ACTOR_CHECKPOINT" \
    --target_dir "$MERGED_OUTPUT_DIR" \
    --hf_model_path "$MODEL_PATH"

echo "Merged model saved to: $MERGED_OUTPUT_DIR"
