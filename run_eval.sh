#!/bin/bash
# =============================================================
# Tapilot-Crossing: Full Pipeline (Prompts -> Inference -> Eval)
# =============================================================
# Usage:
#   bash run_eval.sh
#
# To change models, edit the MODELS array below.
# =============================================================
set -e

# PYTHON="python"
PYTHON="$(which python)"
echo "Using Python: $PYTHON"
$PYTHON -c "import anthropic; print('anthropic import OK')"

# -------------------- CONFIGURATION --------------------
# Add or remove models here
MODELS=(
    "claude-4-6"
    "claude-sonnet-4-5"
    "qwen3-coder"
    "kimi-2.5"
    "minimax-m2.1"
    "glm-4.7"
)

THREADS=8                        # Parallel threads for inference
CODE_GEN_TIMEOUT=60              # Timeout (seconds) per code gen eval
EVAL_WORKERS=8                   # Parallel workers for code gen eval
DATA_DIR="data/dialogue_data"
RESOURCE_DIR="data/resource"
# -------------------------------------------------------

CATEGORIES=(
    "action_analysis.jsonl"
    "action_una.jsonl"
    "action_bg.jsonl"
    "action_plotqa.jsonl"
    "normal.jsonl"
    "private.jsonl"
    "action_correction.jsonl"
    "private_action_correction.jsonl"
)

run_single_model() {
    local MODEL="$1"
    local MODEL_DIR
    MODEL_DIR=$(echo "$MODEL" | sed 's/[-.]/_/g')
    local PROMPT_DIR="output/prompts/${MODEL_DIR}_base"
    local OUTPUT_DIR="output/responses/${MODEL_DIR}_base"

    echo ""
    echo "=========================================="
    echo "  Model: $MODEL"
    echo "=========================================="

    # Step 1: Generate prompts
    echo "[1/3] Generating prompts -> $PROMPT_DIR"
    $PYTHON generate_prompts.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$PROMPT_DIR"

    # Step 2: Inference
    echo "[2/3] Running inference -> $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    for FILE in "${CATEGORIES[@]}"; do
        PROMPT_PATH="${PROMPT_DIR}/${FILE}"
        OUTPUT_PATH="${OUTPUT_DIR}/${FILE}"

        if [ ! -f "$PROMPT_PATH" ]; then
            echo "  Skipping $FILE (prompt file not found)"
            continue
        fi

        # Skip if already completed
        if [ -f "$OUTPUT_PATH" ]; then
            EXPECTED=$(wc -l < "$PROMPT_PATH" | tr -d ' ')
            ACTUAL=$(wc -l < "$OUTPUT_PATH" | tr -d ' ')
            if [ "$ACTUAL" -ge "$EXPECTED" ]; then
                echo "  Skipping $FILE (already complete: $ACTUAL/$EXPECTED)"
                continue
            fi
            echo "  Resuming $FILE ($ACTUAL/$EXPECTED done)"
            START_INDEX="$ACTUAL"
        else
            START_INDEX=0
        fi

        echo "  Processing: $FILE"
        $PYTHON call_api.py \
            --prompt_path "$PROMPT_PATH" \
            --output_path "$OUTPUT_PATH" \
            --model_name "$MODEL" \
            --start_index "$START_INDEX"
    done

    # Step 3: Evaluate
    echo "[3/3] Evaluating..."
    $PYTHON evaluate.py \
        --response_dir "$OUTPUT_DIR" \
        --resource_dir "$RESOURCE_DIR" \
        --model_name "$MODEL_DIR" \
        --code_gen_timeout "$CODE_GEN_TIMEOUT" \
        --workers "$EVAL_WORKERS"

    echo ""
    echo "  Results saved: ${OUTPUT_DIR}/eval_results.json"
}

# ==================== MAIN ====================
echo "####################################################"
echo "#  Tapilot-Crossing Batch Evaluation"
echo "#  Models: ${MODELS[*]}"
echo "#  Threads: $THREADS"
echo "####################################################"

for MODEL in "${MODELS[@]}"; do
    run_single_model "$MODEL" || echo "  WARNING: $MODEL failed, continuing to next model..."
done

# ==================== FINAL SUMMARY ====================
echo ""
echo ""
echo "####################################################"
echo "#              FINAL SUMMARY (All Models)"
echo "####################################################"
echo ""
printf "%-25s %10s %10s %10s %10s\n" "Model" "MC Acc%" "CG Acc%" "Overall%" "Correct/Total"
printf "%-25s %10s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------" "-------------"

for MODEL in "${MODELS[@]}"; do
    MODEL_DIR=$(echo "$MODEL" | sed 's/[-.]/_/g')
    RESULTS_FILE="output/responses/${MODEL_DIR}_base/eval_results.json"

    if [ ! -f "$RESULTS_FILE" ]; then
        printf "%-25s %10s %10s %10s %10s\n" "$MODEL" "N/A" "N/A" "N/A" "N/A"
        continue
    fi

    MC_ACC=$($PYTHON -c "
import json
with open('$RESULTS_FILE') as f:
    r = json.load(f)
mc = r.get('multi_choice_total', {})
print(f\"{mc.get('accuracy', 0):.2f}\")
")
    CG_ACC=$($PYTHON -c "
import json
with open('$RESULTS_FILE') as f:
    r = json.load(f)
cg = r.get('code_gen_total', {})
print(f\"{cg.get('accuracy', 0):.2f}\")
")
    OVERALL=$($PYTHON -c "
import json
with open('$RESULTS_FILE') as f:
    r = json.load(f)
o = r.get('overall', {})
print(f\"{o.get('accuracy', 0):.2f}\")
print(f\"{o.get('correct', 0)}/{o.get('total', 0)}\")
")
    OA_ACC=$(echo "$OVERALL" | head -1)
    OA_FRAC=$(echo "$OVERALL" | tail -1)

    printf "%-25s %10s %10s %10s %10s\n" "$MODEL" "$MC_ACC" "$CG_ACC" "$OA_ACC" "$OA_FRAC"
done

echo ""
echo "####################################################"
echo "#  Done!"
echo "####################################################"
