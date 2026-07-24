#!/bin/bash
# End-to-end test: DPO paired JSONL → convert → tokenize → train (single-GPU LoRA + DeepSpeed LoRA)
#
# Usage: ./run.sh [--model-path PATH] [--max-steps N] ...
# Prerequisite: activate the conda environment or set --python /path/to/python
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults (non-machine-specific) ----
PYTHON=python
DEEPSPEED=deepspeed
MODEL_TYPE=qwen3_5_chatml
MLFLOW_URI=http://localhost:5000
BATCH_MAX_LEN=2048
MAX_STEPS=5
DPO_BETA=0.1

# ---- Required params (no defaults — machine-specific) ----
MODEL_PATH=""

# ---- Parse CLI overrides ----
while [[ $# -gt 0 ]]; do case "$1" in
    --python)         PYTHON="$2"; shift 2 ;;
    --deepspeed)    DEEPSPEED="$2"; shift 2 ;;
    --model-path)     MODEL_PATH="$2"; shift 2 ;;
    --model-type)     MODEL_TYPE="$2"; shift 2 ;;
    --mlflow-uri)     MLFLOW_URI="$2"; shift 2 ;;
    --batch-max-len)  BATCH_MAX_LEN="$2"; shift 2 ;;
    --max-steps)      MAX_STEPS="$2"; shift 2 ;;
    --dpo-beta)       DPO_BETA="$2"; shift 2 ;;
    -h|--help)
        echo "Usage: $0 --model-path PATH [OPTIONS]"
        echo "Required:"
        echo "  --model-path PATH     Model directory"
        echo "Options:"
        echo "  --python PATH         Python binary (default: $PYTHON)"
        echo "  --deepspeed PATH      DeepSpeed binary (default: $DEEPSPEED)"
        echo "  --model-type TYPE     Model type for config (default: $MODEL_TYPE)"
        echo "  --mlflow-uri URI      MLflow tracking URI (default: $MLFLOW_URI)"
        echo "  --batch-max-len N     Batch max length (default: $BATCH_MAX_LEN)"
        echo "  --max-steps N         Training steps (default: $MAX_STEPS)"
        echo "  --dpo-beta FLOAT      DPO temperature (default: $DPO_BETA)"
        exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
esac; done

# ---- Validate required params ----
if [[ -z "$MODEL_PATH" ]]; then echo "ERROR: --model-path is required"; exit 1; fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== DPO End-to-End Test ==="
echo "Model:  $MODEL_PATH"
echo "Steps:  $MAX_STEPS"
echo "Repo:   $REPO_ROOT"
echo ""

# ---- Clean & prep ----
rm -rf converted pretokenized output
mkdir -p converted pretokenized output

# ---- Step 1: Convert ----
echo "=== Step 1/7: Convert OpenAI paired JSONL → Conversation format ==="
$PYTHON -m ochat.data.convert_dataset_dpo \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files data_dpo.jsonl \
    --out-file converted/train_dpo.jsonl
echo "  → $(wc -l < converted/train_dpo.jsonl) pairs"

# ---- Step 2: Tokenize (skip ref log-probs for speed) ----
echo ""
echo "=== Step 2/7: Tokenize → parquet ==="
$PYTHON -m ochat.data.generate_dpo_dataset \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files converted/train_dpo.jsonl \
    --out-prefix pretokenized/dpo_data \
    --data-length-multiple-of 64 \
    --eval-ratio 0.2 \
    --num-splits 1 \
    --max-jobs 1
echo "  → train: $(ls -lh pretokenized/dpo_data.train.parquet)"
echo "  → eval:  $(ls -lh pretokenized/dpo_data.eval.parquet)"

# ---- Step 3: Cache ref log-probs (standalone, enables full FT) ----
echo ""
echo "=== Step 3/7: Cache reference log-probs ==="
$PYTHON -m ochat.data.cache_ref_logps \
    --data-prefix pretokenized/dpo_data \
    --model-path "$MODEL_PATH"
echo "  → train cache: $(ls -lh pretokenized/dpo_data.train.ref_logps_cache.npz)"
echo "  → eval cache:  $(ls -lh pretokenized/dpo_data.eval.ref_logps_cache.npz)"

# ---- Step 4: Single-GPU full FT ----
echo ""
echo "=== Step 4/7: Single-GPU DPO full FT (${MAX_STEPS} steps) ==="
rm -rf output/dpo_full_ft
$PYTHON -m ochat.training_dpo.train_single \
    --local_rank 0 \
    --model_path "$MODEL_PATH" \
    --data_prefix pretokenized/dpo_data \
    --save_path output/dpo_full_ft \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --dpo_beta "$DPO_BETA" \
    --experiment_name e2e_dpo --run_name dpo_full_ft \
    --tracking_uri "$MLFLOW_URI"
echo "  → output/dpo_full_ft/"

# ---- Step 5: Single-GPU LoRA ----
echo ""
echo "=== Step 5/7: Single-GPU DPO LoRA (${MAX_STEPS} steps) ==="
rm -rf output/dpo_lora
$PYTHON -m ochat.training_dpo.train_single \
    --local_rank 0 \
    --model_path "$MODEL_PATH" \
    --data_prefix pretokenized/dpo_data \
    --save_path output/dpo_lora \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --use_lora \
    --dpo_beta "$DPO_BETA" \
    --experiment_name e2e_dpo --run_name dpo_lora \
    --tracking_uri "$MLFLOW_URI"
echo "  → output/dpo_lora/"

# ---- Step 6: DeepSpeed full FT ----
echo ""
echo "=== Step 6/7: DeepSpeed DPO full FT (${MAX_STEPS} steps) ==="
rm -rf output/dpo_deepspeed_ft
DS_CONFIG="$REPO_ROOT/ochat/deepspeed_config/deepspeed_config.json"
(cd "$REPO_ROOT" && $DEEPSPEED --num_gpus 1 \
    --module ochat.training_dpo.train \
    --model_path "$MODEL_PATH" \
    --data_prefix "$SCRIPT_DIR/pretokenized/dpo_data" \
    --save_path "$SCRIPT_DIR/output/dpo_deepspeed_ft" \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --dpo_beta "$DPO_BETA" \
    --deepspeed --deepspeed_config "$DS_CONFIG" \
    --experiment_name e2e_dpo --run_name dpo_deepspeed_ft \
    --tracking_uri "$MLFLOW_URI")
echo "  → output/dpo_deepspeed_ft/"

# ---- Step 7: DeepSpeed LoRA ----
echo ""
echo "=== Step 7/7: DeepSpeed DPO LoRA (${MAX_STEPS} steps) ==="
rm -rf output/dpo_deepspeed_lora
(cd "$REPO_ROOT" && $DEEPSPEED --num_gpus 1 \
    --module ochat.training_dpo.train \
    --model_path "$MODEL_PATH" \
    --data_prefix "$SCRIPT_DIR/pretokenized/dpo_data" \
    --save_path "$SCRIPT_DIR/output/dpo_deepspeed_lora" \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --use_lora \
    --dpo_beta "$DPO_BETA" \
    --deepspeed --deepspeed_config "$DS_CONFIG" \
    --experiment_name e2e_dpo --run_name dpo_deepspeed_lora \
    --tracking_uri "$MLFLOW_URI")
echo "  → output/dpo_deepspeed_lora/"

# ---- Done ----
echo ""
echo "=== All done ==="
echo "DPO full FT:          output/dpo_full_ft/"
echo "DPO LoRA:             output/dpo_lora/"
echo "DPO DeepSpeed FT:     output/dpo_deepspeed_ft/"
echo "DPO DeepSpeed LoRA:   output/dpo_deepspeed_lora/"
echo "MLflow:               $MLFLOW_URI (experiment: e2e_dpo)"
