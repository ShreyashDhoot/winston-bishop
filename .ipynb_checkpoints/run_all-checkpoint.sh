#!/usr/bin/env bash
# run_all.sh  —  all attacks per model → eval → next model, with resume
# Usage: bash run_all.sh --dataset path/to/dataset.json [--limit 25] [--seed 0] [--no-tipai]

set -euo pipefail

DATASET=""
LIMIT=25
SEED=0
TIPAI_FLAG=""
EXTRA_FLAGS=""
PROGRESS_FILE=".run_all_progress"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)    DATASET="$2";    shift 2 ;;
    --limit)      LIMIT="$2";      shift 2 ;;
    --seed)       SEED="$2";       shift 2 ;;
    --no-tipai)   TIPAI_FLAG="--no-tipai"; shift ;;
    --no-shuffle) EXTRA_FLAGS="$EXTRA_FLAGS --no-shuffle"; shift ;;
    --reset)      rm -f "$PROGRESS_FILE"; echo "[Resume] Progress cleared."; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "ERROR: --dataset <path> is required"; exit 1
fi

MODELS=("SD 1.5" "SDXL" "SD 3.5 Med" "SD 3.5 Turbo" "Flux.1")
ATTACKS=("DACA" "PGJ" "MMA" "RingABell" "SneakPrompt")

# ── helpers ───────────────────────────────────────────────────────────────────
mark_done()  { echo "$1" >> "$PROGRESS_FILE"; }
is_done()    { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }

# ── main loop ─────────────────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  MODEL: $MODEL"
  echo "════════════════════════════════════════════════════════════"

  for ATTACK in "${ATTACKS[@]}"; do
    KEY="attack::${MODEL}::${ATTACK}"

    if is_done "$KEY"; then
      echo "  [skip] $ATTACK already completed — resuming"
      continue
    fi

    echo ""
    echo "  ── Attack: $ATTACK ──────────────────────────────────────"
    python jailbreak_tispa.py \
      --dataset  "$DATASET" \
      --type     "$ATTACK" \
      --model    "$MODEL" \
      --limit    "$LIMIT" \
      --seed     "$SEED" \
      $TIPAI_FLAG \
      $EXTRA_FLAGS

    mark_done "$KEY"
  done

  EVAL_KEY="eval::${MODEL}"
  if is_done "$EVAL_KEY"; then
    echo "  [skip] Eval for $MODEL already done — resuming"
  else
    echo ""
    echo "  ── Eval after all attacks for: $MODEL ──────────────────"
    python evaluate_asr.py
    mark_done "$EVAL_KEY"
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "════════════════════════════════════════════════════════════"

# Clean up progress file on clean completion
rm -f "$PROGRESS_FILE"