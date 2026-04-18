#!/bin/bash
# ReQueR RL training (GRPO) with Adaptive Solver Hierarchy and perplexity-based
# leak judge. Trains a Refiner (Qwen3-4B by default) against a pool of frozen
# Solvers so that the resulting policy generalizes one-to-many to unseen models.
#
# Prereqs:
#   - An SFT-initialized Refiner checkpoint (see rephrase_cold_start_sft/)
#   - Solver weights (Qwen3-0.6B / 1.7B / 4B) under $MODEL_DIR
#   - Prepared parquet training/validation data (see scripts/prepare_mix_rephrase_dataset.py
#     and scripts/prepare_math500_test_dataset.py)
#   - Optional: running perplexity-judge service (see ../perplexity-judge/)

set -eux

# --- Required paths (user must set these) ----------------------------------
: "${REQUER_ROOT:?set REQUER_ROOT to the repo root (e.g. export REQUER_ROOT=\$(pwd)/..)}"
: "${MODEL_DIR:?set MODEL_DIR to the directory containing Qwen3-*B weights}"
: "${ACTOR_MODEL_PATH:?set ACTOR_MODEL_PATH to the SFT-initialized Refiner checkpoint}"

# --- Data ------------------------------------------------------------------
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${REQUER_ROOT}/verl/data/mix/simple_template/mix_rephrase_train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${REQUER_ROOT}/verl/data/mix/simple_template/math500_test.parquet}"

# --- Solver pool (ASH) -----------------------------------------------------
export REPHRASE_SOLVER_ROOT="$MODEL_DIR"
export REPHRASE_SOLVER_MODELS="${REPHRASE_SOLVER_MODELS:-Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B}"
export REPHRASE_SOLVER_VISIBLE_DEVICES="${REPHRASE_SOLVER_VISIBLE_DEVICES:-4,5,6}"
export REPHRASE_SOLVER_DEVICE_MAP="${REPHRASE_SOLVER_DEVICE_MAP:-4,5,6}"
export RAY_CGRAPH_get_timeout=600

# --- Perplexity leak judge -------------------------------------------------
export ENABLE_PERPLEXITY_LEAK_JUDGE="${ENABLE_PERPLEXITY_LEAK_JUDGE:-true}"
export PERPLEXITY_LEAK_THRESHOLD="${PERPLEXITY_LEAK_THRESHOLD:-5.0}"
export LEAK_JUDGE_MODEL_PATH="${LEAK_JUDGE_MODEL_PATH:-$MODEL_DIR/Qwen3-0.6B}"
export LEAK_JUDGE_DEVICE="${LEAK_JUDGE_DEVICE:-cuda:7}"

# --- Experiment / logging --------------------------------------------------
PROJECT_NAME="${PROJECT_NAME:-requer_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-refiner_qwen3_4b_ash_leak5.0}"

# WandB (set WANDB_API_KEY externally if you want online sync; leave unset
# or use WANDB_MODE=offline otherwise).
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_ID="$EXPERIMENT_NAME"

# --- GRPO ------------------------------------------------------------------
NUM_VARIANTS_PER_PROMPT="${NUM_VARIANTS_PER_PROMPT:-8}"

# --- vLLM offline inference (for batched Solver rollout) -------------------
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_DIR/Qwen3-1.7B}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-1024}"
export USE_VLLM_OFFLINE="${USE_VLLM_OFFLINE:-true}"

# --- API fallback (unused when USE_VLLM_OFFLINE=true) ----------------------
# export REWARD_API_URL="..."
# export REWARD_API_MODEL="..."
# export REWARD_API_MAX_CONCURRENCY="256"

# --- Sanity checks ---------------------------------------------------------
[ -f "$TRAIN_DATA_PATH" ] || { echo "ERROR: training data not found at $TRAIN_DATA_PATH"; echo "Run scripts/prepare_mix_rephrase_dataset.py first."; exit 1; }
[ -f "$VAL_DATA_PATH" ]   || { echo "ERROR: validation data not found at $VAL_DATA_PATH"; echo "Run scripts/prepare_math500_test_dataset.py first."; exit 1; }

if [ "$USE_VLLM_OFFLINE" = "true" ] && [ -z "$VLLM_MODEL_PATH" ]; then
    echo "ERROR: VLLM_MODEL_PATH is not set for vLLM offline mode." >&2
    exit 1
fi
if [ "$USE_VLLM_OFFLINE" != "true" ] && [ -z "${REWARD_API_URL:-}" ]; then
    echo "ERROR: REWARD_API_URL is not set for API mode." >&2
    exit 1
fi

echo "=== ReQueR RL Training ==="
echo "Project    : $PROJECT_NAME"
echo "Experiment : $EXPERIMENT_NAME"
echo "Refiner    : $ACTOR_MODEL_PATH"
echo "Solvers    : $REPHRASE_SOLVER_MODELS (root=$REPHRASE_SOLVER_ROOT)"
echo "Variants/prompt: $NUM_VARIANTS_PER_PROMPT"
echo "==========================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_DATA_PATH']" \
    data.val_files="['$VAL_DATA_PATH']" \
    +data.data_source_type='api_based_reward' \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$NUM_VARIANTS_PER_PROMPT" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "$@"
