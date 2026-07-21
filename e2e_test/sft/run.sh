#!/bin/bash
# End-to-end test: OpenAI JSONL → convert → tokenize → train (4 modes)
#
# Usage: ./run.sh [--model-path PATH] [--python PATH] [--deepspeed PATH] ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults (non-machine-specific) ----
PYTHON=python
DEEPSPEED=deepspeed
MODEL_TYPE=qwen3_5_chatml
MLFLOW_URI=http://localhost:5000
DS_CONFIG=ochat/deepspeed_config/deepspeed_config.json
BATCH_MAX_LEN=2048
MAX_STEPS=5

# ---- Required params (no defaults — machine-specific) ----
MODEL_PATH=""

# ---- Parse CLI overrides ----
while [[ $# -gt 0 ]]; do case "$1" in
    --python)       PYTHON="$2"; shift 2 ;;
    --deepspeed)    DEEPSPEED="$2"; shift 2 ;;
    --model-path)   MODEL_PATH="$2"; shift 2 ;;
    --model-type)   MODEL_TYPE="$2"; shift 2 ;;
    --mlflow-uri)   MLFLOW_URI="$2"; shift 2 ;;
    --ds-config)    DS_CONFIG="$2"; shift 2 ;;
    --batch-max-len) BATCH_MAX_LEN="$2"; shift 2 ;;
    --max-steps)    MAX_STEPS="$2"; shift 2 ;;
    -h|--help)
        echo "Usage: $0 --model-path PATH [OPTIONS]"
        echo "Required:"
        echo "  --model-path PATH     Model directory"
        echo "Options:"
        echo "  --python PATH         Python binary (default: $PYTHON)"
        echo "  --deepspeed PATH      DeepSpeed binary (default: $DEEPSPEED)"
        echo "  --model-type TYPE     Model type for config (default: $MODEL_TYPE)"
        echo "  --mlflow-uri URI      MLflow tracking URI (default: $MLFLOW_URI)"
        echo "  --ds-config PATH      DeepSpeed config JSON (default: $DS_CONFIG)"
        echo "  --batch-max-len N     Batch max length (default: $BATCH_MAX_LEN)"
        echo "  --max-steps N         Training steps per mode (default: $MAX_STEPS)"
        exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
esac; done

# ---- Validate required params ----
if [[ -z "$MODEL_PATH" ]]; then echo "ERROR: --model-path is required"; exit 1; fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- Clean & prep ----
rm -rf converted pretokenized output
mkdir -p converted pretokenized output

# ---- Step 1: Convert ----
echo "=== Step 1/6: Convert OpenAI JSONL → OpenChat format ==="
$PYTHON -m ochat.data.convert_dataset \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files data.jsonl \
    --out-file converted/train.jsonl
echo "  → $(wc -l < converted/train.jsonl) conversations"

# ---- Step 2: Tokenize ----
echo ""
echo "=== Step 2/6: Tokenize → parquet ==="
$PYTHON -m ochat.data.generate_dataset \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files converted/train.jsonl \
    --out-prefix pretokenized/data \
    --data-length-multiple-of 64 \
    --eval-ratio 0.2 \
    --num-splits 1 \
    --max-jobs 1
echo "  → train: $(ls -lh pretokenized/data.train.parquet)"
echo "  → eval:  $(ls -lh pretokenized/data.eval.parquet)"

# ---- Step 3: Single-GPU full FT ----
echo ""
echo "=== Step 3/6: Single-GPU full fine-tuning (${MAX_STEPS} steps) ==="
rm -rf output/single_full_ft
$PYTHON -m ochat.training_sft.train_single \
    --model_path "$MODEL_PATH" \
    --data_prefix pretokenized/data \
    --save_path output/single_full_ft \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --experiment_name e2e_sft --run_name single_full_ft \
    --tracking_uri "$MLFLOW_URI"
echo "  → output/single_full_ft/"

# ---- Step 4: Single-GPU LoRA ----
echo ""
echo "=== Step 4/6: Single-GPU LoRA (${MAX_STEPS} steps) ==="
rm -rf output/single_lora
$PYTHON -m ochat.training_sft.train_single \
    --model_path "$MODEL_PATH" \
    --data_prefix pretokenized/data \
    --save_path output/single_lora \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --use_lora \
    --experiment_name e2e_sft --run_name single_lora \
    --tracking_uri "$MLFLOW_URI"
echo "  → output/single_lora/"

# ---- Step 5: DeepSpeed full FT (1 GPU) ----
echo ""
echo "=== Step 5/6: DeepSpeed full fine-tuning (${MAX_STEPS} steps) ==="
rm -rf output/deepspeed_full_ft
(cd /home/fahadh/research-openchat/openchat && $DEEPSPEED --num_gpus 1 \
    --module ochat.training_sft.train \
    --model_path "$MODEL_PATH" \
    --data_prefix "$SCRIPT_DIR/pretokenized/data" \
    --save_path "$SCRIPT_DIR/output/deepspeed_full_ft" \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --deepspeed --deepspeed_config "$DS_CONFIG" \
    --experiment_name e2e_sft --run_name deepspeed_full_ft \
    --tracking_uri "$MLFLOW_URI")
echo "  → output/deepspeed_full_ft/"

# ---- Step 6: DeepSpeed LoRA (1 GPU) ----
echo ""
echo "=== Step 6/6: DeepSpeed LoRA (${MAX_STEPS} steps) ==="
rm -rf output/deepspeed_lora
(cd /home/fahadh/research-openchat/openchat && $DEEPSPEED --num_gpus 1 \
    --module ochat.training_sft.train \
    --model_path "$MODEL_PATH" \
    --data_prefix "$SCRIPT_DIR/pretokenized/data" \
    --save_path "$SCRIPT_DIR/output/deepspeed_lora" \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --use_lora \
    --deepspeed --deepspeed_config "$DS_CONFIG" \
    --experiment_name e2e_sft --run_name deepspeed_lora \
    --tracking_uri "$MLFLOW_URI")
echo "  → output/deepspeed_lora/"

# ---- Done ----
echo ""
echo "=== All done ==="
echo "Single-GPU full FT:    output/single_full_ft/"
echo "Single-GPU LoRA:       output/single_lora/"
echo "DeepSpeed full FT:     output/deepspeed_full_ft/"
echo "DeepSpeed LoRA:        output/deepspeed_lora/"
echo "MLflow:                $MLFLOW_URI (experiment: e2e_sft)"
