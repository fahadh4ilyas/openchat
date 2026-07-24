#!/bin/bash
# End-to-end test: KTO unpaired JSONL → convert → tokenize → train (single-GPU + DeepSpeed)
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
KTO_BETA=0.1

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
    --kto-beta)       KTO_BETA="$2"; shift 2 ;;
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
        echo "  --kto-beta FLOAT      KTO temperature (default: $KTO_BETA)"
        exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
esac; done

# ---- Validate required params ----
if [[ -z "$MODEL_PATH" ]]; then echo "ERROR: --model-path is required"; exit 1; fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== KTO End-to-End Test ==="
echo "Model:  $MODEL_PATH"
echo "Steps:  $MAX_STEPS"
echo "Repo:   $REPO_ROOT"
echo ""

# ---- Clean & prep ----
rm -rf converted pretokenized output
mkdir -p converted pretokenized output

# ---- Step 1: Convert OpenAI JSONL → Conversation format ----
echo "=== Step 1/5: Convert OpenAI JSONL → Conversation format ==="
$PYTHON -m ochat.data.convert_dataset \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files data_kto.jsonl \
    --out-file converted/train_kto.jsonl
echo "  → $(wc -l < converted/train_kto.jsonl) examples ($(grep -c '"label":true' converted/train_kto.jsonl) desirable, $(grep -c '"label":false' converted/train_kto.jsonl) undesirable)"

# ---- Step 2: Tokenize ----
echo ""
echo "=== Step 2/4: Tokenize → parquet (--kto) ==="
$PYTHON -m ochat.data.generate_dataset \
    --model-type "$MODEL_TYPE" \
    --model-path "$MODEL_PATH" \
    --in-files converted/train_kto.jsonl \
    --out-prefix pretokenized/kto_data \
    --kto \
    --data-length-multiple-of 64 \
    --eval-ratio 0.2 \
    --num-splits 1 \
    --max-jobs 1
echo "  → train: $(ls -lh pretokenized/kto_data.train.parquet)"
echo "  → eval:  $(ls -lh pretokenized/kto_data.eval.parquet)"

# ---- Step 4: Single-GPU ----
echo ""
echo "=== Step 3/4: Single-GPU KTO (${MAX_STEPS} steps) ==="
rm -rf output/kto_single
$PYTHON -m ochat.training_kto.train_single \
    --local_rank 0 \
    --model_path "$MODEL_PATH" \
    --data_prefix pretokenized/kto_data \
    --save_path output/kto_single \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --kto_beta "$KTO_BETA" \
    --experiment_name e2e_kto --run_name kto_single \
    --tracking_uri "$MLFLOW_URI"
echo "  → output/kto_single/"

# ---- Step 5: DeepSpeed ----
echo ""
echo "=== Step 4/4: DeepSpeed KTO (${MAX_STEPS} steps) ==="
rm -rf output/kto_deepspeed
DS_CONFIG="$REPO_ROOT/ochat/deepspeed_config/deepspeed_config.json"
(cd "$REPO_ROOT" && $DEEPSPEED --num_gpus 1 \
    --module ochat.training_kto.train \
    --model_path "$MODEL_PATH" \
    --data_prefix "$SCRIPT_DIR/pretokenized/kto_data" \
    --save_path "$SCRIPT_DIR/output/kto_deepspeed" \
    --batch_max_len "$BATCH_MAX_LEN" \
    --epochs 1 --max_steps "$MAX_STEPS" \
    --kto_beta "$KTO_BETA" \
    --deepspeed --deepspeed_config "$DS_CONFIG" \
    --experiment_name e2e_kto --run_name kto_deepspeed \
    --tracking_uri "$MLFLOW_URI")
echo "  → output/kto_deepspeed/"

# ---- Done ----
echo ""
echo "=== All done ==="
echo "KTO single-GPU:       output/kto_single/"
echo "KTO DeepSpeed:        output/kto_deepspeed/"
echo "MLflow:               $MLFLOW_URI (experiment: e2e_kto)"
