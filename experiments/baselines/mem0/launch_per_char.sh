#!/usr/bin/env bash
# launch_per_char.sh
# Run 11 isolated per-character ingest processes.
#
# Each character gets:
#   - own qdrant path:    .mem0/qdrant_CHAR_XX/
#   - own ingested log:   .mem0/ingested_CHAR_XX.jsonl
#   - own snapshot dir:   .mem0/snapshots/CHAR_XX/  (already per-char from ingest script)
#   - own log file:       .mem0/logs/CHAR_XX.log
#
# Concurrency is capped via xargs -P. Default = 4 (tune for LLM/embed rate limits).
#
# Requires: ingest_memories.py to honor $MEM0_CHAR_TAG (see "change 1" in chat).
#
# Usage:
#   bash launch_per_char.sh                # all 11 chars, 4 concurrent
#   PARALLEL=2 bash launch_per_char.sh     # tune concurrency
#   CHARS="CHAR_01 CHAR_02" bash launch_per_char.sh   # subset

set -u  # do NOT set -e: a single char failing should not stop the others

PARALLEL="${PARALLEL:-4}"
CHARS="${CHARS:-CHAR_01 CHAR_02 CHAR_03 CHAR_04 CHAR_05 CHAR_06 CHAR_07 CHAR_08 CHAR_09 CHAR_10 CHAR_11}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$ROOT/.mem0/logs"
mkdir -p "$LOGDIR"

# Activate venv
source "$ROOT/.venv/bin/activate" || { echo "venv activation failed"; exit 1; }

# Disable mem0 telemetry to avoid the global ~/.mem0/migrations_qdrant lock
export MEM0_TELEMETRY=False

echo "launching: chars=[$CHARS]  parallel=$PARALLEL"
echo "logs -> $LOGDIR/CHAR_XX.log"
echo

# xargs runs N processes concurrently. Each process is fully isolated:
# its own MEM0_CHAR_TAG drives ingest_memories.py to use its own qdrant +
# ingested log paths.
printf "%s\n" $CHARS | xargs -n 1 -P "$PARALLEL" -I{} bash -c '
  CID="$1"
  STARTED="$(date +%T)"
  echo "[$STARTED] start $CID"

  MEM0_CHAR_TAG="$CID" \
    python ingest_memories.py --char "$CID" --snapshot-by-age \
    > "'"$LOGDIR"'/$CID.log" 2>&1
  RC=$?

  ENDED="$(date +%T)"
  if [ $RC -eq 0 ]; then
    echo "[$ENDED] done  $CID  ok"
  else
    echo "[$ENDED] FAIL  $CID  exit=$RC  (see '"$LOGDIR"'/$CID.log)"
  fi
' _ {}

echo
echo "all launched processes have exited. summary:"
for c in $CHARS; do
  log="$LOGDIR/$c.log"
  if [ -f "$log" ]; then
    last=$(tail -3 "$log" | tr '\n' ' ')
    echo "  $c: $last"
  else
    echo "  $c: (no log)"
  fi
done
