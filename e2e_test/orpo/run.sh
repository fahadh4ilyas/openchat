#!/bin/bash
# End-to-end test: ORPO paired JSONL → convert → tokenize → train (full FT + LoRA)
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
ORPO_BETA=0.1

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
    --orpo-beta)      ORPO_BETA="$2"; shift 2 ;;
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
        echo "  --orpo-beta FLOAT     ORPO temperature (default: $ORPO_BETA)"
        exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
esac; done

# ---- Validate required params ----
if [[ -z "$MODEL_PATH" ]]; then echo "ERROR: --model-path is required"; exit 1; fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== ORPO End-to-End Test ==="
echo "Model:  $MODEL_PATH"
echo "Steps:  $MAX_STEPS"
echo "Repo:   $REPO_ROOT"
echo ""

# ---- Clean & prep ----
rm -rf converted pretokenized output
mkdir -p converted pretokenized output

# ---- Step 1: Convert (ORPO uses same format as DPO) ----
echo "=== Step 1/6: Convert OpenAI paired JSONL → Conversation format ==="
$PYTHON -m ochat.data.convert_dataset --train-type orpo \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files data_orpo.jsonl \
    --out-file converted/train_orpo.jsonl
echo "  → $(wc -l < converted/train_orpo.jsonl) pairs"

# ---- Step 2: Tokenize (no ref log-probs needed for ORPO) ----
echo ""
echo "=== Step 2/6: Tokenize → parquet ==="
$PYTHON -m ochat.data.generate_dataset --train-type orpo \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files converted/train_orpo.jsonl \
    --out-prefix pretokenized/orpo_data \
    --data-length-multiple-of 64 \
    --eval-ratio 0.2 \
    --num-splits 1 \
    --max-jobs 1
echo "  → train: $(ls -lh pretokenized/orpo_data.train.parquet)"
echo "  → eval:  $(ls -lh pretokenized/orpo_data.eval.parquet)"

# ---- Step 3: Single-GPU full FT ----
echo ""
echo "=== Step 3/6: Single-GPU ORPO full FT (${MAX_STEPS} steps) ==="
rm -rf output/orpo_full_ft
$PYTHON -m ochat.training_orpo.train_single \
    --local-rank 0 \
    --model-path "$MODEL_PATH" \
    --data-prefix pretokenized/orpo_data \
    --save-path output/orpo_full_ft \
    --batch-max-len "$BATCH_MAX_LEN" \
    --epochs 1 --max-steps "$MAX_STEPS" \
    --orpo-beta "$ORPO_BETA" \
    --experiment-name e2e_orpo --run-name orpo_full_ft \
    --tracking-uri "$MLFLOW_URI"
echo "  → output/orpo_full_ft/"

# ---- Step 4: Single-GPU LoRA ----
echo ""
echo "=== Step 4/6: Single-GPU ORPO LoRA (${MAX_STEPS} steps) ==="
rm -rf output/orpo_lora
$PYTHON -m ochat.training_orpo.train_single \
    --local-rank 0 \
    --model-path "$MODEL_PATH" \
    --data-prefix pretokenized/orpo_data \
    --save-path output/orpo_lora \
    --batch-max-len "$BATCH_MAX_LEN" \
    --epochs 1 --max-steps "$MAX_STEPS" \
    --use-lora \
    --orpo-beta "$ORPO_BETA" \
    --experiment-name e2e_orpo --run-name orpo_lora \
    --tracking-uri "$MLFLOW_URI"
echo "  → output/orpo_lora/"

# ---- Step 5: DeepSpeed full FT ----
echo ""
echo "=== Step 5/6: DeepSpeed ORPO full FT (${MAX_STEPS} steps) ==="
rm -rf output/orpo_deepspeed_ft
DS_CONFIG="$REPO_ROOT/ochat/deepspeed_config/deepspeed_config.json"
(cd "$REPO_ROOT" && $DEEPSPEED --num-gpus 1 \
    --module ochat.training_orpo.train \
    --model-path "$MODEL_PATH" \
    --data-prefix "$SCRIPT_DIR/pretokenized/orpo_data" \
    --save-path "$SCRIPT_DIR/output/orpo_deepspeed_ft" \
    --batch-max-len "$BATCH_MAX_LEN" \
    --epochs 1 --max-steps "$MAX_STEPS" \
    --orpo-beta "$ORPO_BETA" \
    --deepspeed --deepspeed-config "$DS_CONFIG" \
    --experiment-name e2e_orpo --run-name orpo_deepspeed_ft \
    --tracking-uri "$MLFLOW_URI")
echo "  → output/orpo_deepspeed_ft/"

# ---- Step 6: DeepSpeed LoRA ----
echo ""
echo "=== Step 6/6: DeepSpeed ORPO LoRA (${MAX_STEPS} steps) ==="
rm -rf output/orpo_deepspeed_lora
(cd "$REPO_ROOT" && $DEEPSPEED --num-gpus 1 \
    --module ochat.training_orpo.train \
    --model-path "$MODEL_PATH" \
    --data-prefix "$SCRIPT_DIR/pretokenized/orpo_data" \
    --save-path "$SCRIPT_DIR/output/orpo_deepspeed_lora" \
    --batch-max-len "$BATCH_MAX_LEN" \
    --epochs 1 --max-steps "$MAX_STEPS" \
    --use-lora \
    --orpo-beta "$ORPO_BETA" \
    --deepspeed --deepspeed-config "$DS_CONFIG" \
    --experiment-name e2e_orpo --run-name orpo_deepspeed_lora \
    --tracking-uri "$MLFLOW_URI")
echo "  → output/orpo_deepspeed_lora/"

# ---- Done ----
echo ""
echo "=== All done ==="
echo "ORPO full FT:          output/orpo_full_ft/"
echo "ORPO LoRA:             output/orpo_lora/"
echo "ORPO DeepSpeed FT:     output/orpo_deepspeed_ft/"
echo "ORPO DeepSpeed LoRA:   output/orpo_deepspeed_lora/"
echo "MLflow:                $MLFLOW_URI (experiment: e2e_orpo)"
