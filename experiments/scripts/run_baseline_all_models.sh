#!/bin/bash
# Run the baseline_model_only experiment across all 7 models.
# Usage: bash run_baseline_all_models.sh [workers]
# Defaults to 16 workers per model, run sequentially to avoid hammering the same gateway.

set -u
cd "$(dirname "$0")"

WORKERS="${1:-16}"
LOG_DIR="results/baseline_model_only/_logs"
mkdir -p "$LOG_DIR"

# The 7 models aligned with mem0/naive_rag
MODELS=(
  "gemini-3.1-pro-preview"
  "gemini-3-flash-preview"
  "claude-haiku-4-5"
  "gpt-5.4-mini"
  "qwen3.5-397b-a17b"
  "qwen3.5-122b-a10b"
  "qwen3.5-35b-a3b"
)

# Use the SJTU gateway from .env
export API_KEY="${API_KEY:-REMOVED_LEAKED_KEY}"
export API_BASE="${API_BASE:-https://llm-sjtu.multiego.me/v1}"
export TIMEOUT="${TIMEOUT:-180}"

echo "====================================================="
echo "BASELINE_MODEL_ONLY: 7 models"
echo "  api_base: $API_BASE"
echo "  workers : $WORKERS each"
echo "  log_dir : $LOG_DIR"
echo "====================================================="

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL="${MODEL//\//_}"
  LOG="$LOG_DIR/${SAFE_MODEL}.log"
  echo
  echo ">>> [$(date '+%H:%M:%S')] running model=$MODEL  -> $LOG"
  python3 run_baseline_model_only.py \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --temperature 0 \
    > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "<<< [$(date '+%H:%M:%S')] model=$MODEL FAILED (rc=$RC), see $LOG"
  else
    # Print a one-line summary
    SUMMARY="results/baseline_model_only/${SAFE_MODEL}/summary_baseline.json"
    if [ -f "$SUMMARY" ]; then
      python3 -c "
import json
d = json.load(open('$SUMMARY'))
print(f'<<< [$MODEL] total={d[\"total_questions\"]} ok={d[\"ok\"]} correct={d[\"correct\"]} acc={d[\"accuracy_overall\"]:.4f}  letter_dist={d[\"letter_distribution\"]}')
"
    fi
  fi
done

echo
echo "====================================================="
echo "ALL DONE — summary table:"
python3 -c "
import json, glob
for f in sorted(glob.glob('results/baseline_model_only/*/summary_baseline.json')):
    d = json.load(open(f))
    print(f'  {d[\"model\"]:<26}  acc={d[\"accuracy_overall\"]:.4f}  ok={d[\"ok\"]}/{d[\"total_questions\"]}  ld={d[\"letter_distribution\"]}')
"
