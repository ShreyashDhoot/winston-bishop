#!/usr/bin/env bash
# ==============================================================================
# run_all_experiments.sh
# ───────────────────────
# Master Orchestrator for GuardPaint (TiPAI-TSPO) Next-Cycle Submission Revisions.
#
# Runs:
#   1. Baseline Defenses (SLD, Post-Hoc Detect+Regen, ESD, SafeGen, Latent Guard)
#   2. Component Ablations (Auditor-Refusal, SFT-Only, No-Tournament, Random-Knobs)
#   3. Auditor External Validity Evaluation (UnsafeBench, T2ISafety)
#   4. Multi-Task Loss-Weight Sensitivity Sweep (±50%)
#   5. Evaluates all results with robust Qwen verdict parser
#   6. Compiles all publication-ready Markdown/LaTeX tables and PDF/PNG figures
#
# Usage:
#   bash new-submission-code/run_all_experiments.sh --all
#   bash new-submission-code/run_all_experiments.sh --tables-only
#   bash new-submission-code/run_all_experiments.sh --baselines --limit 50
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default Arguments
DATASET="JailbreakDiffusionBench/data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json"
LIMIT=100
MODE="all"
PROGRESS_FILE="$SCRIPT_DIR/.submission_progress"
LOG_FILE="$SCRIPT_DIR/experiments.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)         MODE="all"; shift ;;
    --baselines)   MODE="baselines"; shift ;;
    --ablations)   MODE="ablations"; shift ;;
    --tables-only) MODE="tables"; shift ;;
    --dataset)     DATASET="$2"; shift 2 ;;
    --limit)       LIMIT="$2"; shift 2 ;;
    --reset)       rm -f "$PROGRESS_FILE"; echo "[Progress] Cleared previous checkpoints."; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Ensure SSL bypass for server environments with proxy inspection
export HF_HUB_DISABLE_SSL_VERIFY=1
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""

mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/tables/output"

mark_done() { echo "$1" >> "$PROGRESS_FILE"; }
is_done()   { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

log "======================================================================"
log "  GuardPaint (TiPAI-TSPO) Next-Cycle Revision Suite"
log "  Mode: $MODE | Dataset: $DATASET | Limit: $LIMIT"
log "======================================================================"

# ── Phase 1: Baseline Defenses ───────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "baselines" ]]; then
  BASELINES=("sld" "post_hoc" "esd" "safegen" "latent_guard")
  MODELS=("SD 1.5" "SDXL" "Flux.1")

  log "--> Starting Phase 1: Baseline Defenses Comparison"
  for B in "${BASELINES[@]}"; do
    for M in "${MODELS[@]}"; do
      KEY="baseline::${B}::${M}"
      if is_done "$KEY"; then
        log "  [skip] Baseline $B on $M already completed — skipping."
        continue
      fi

      log "  [Run] Executing Baseline: $B on Model: $M"
      python "$SCRIPT_DIR/baselines/run_baselines.py" \
        --dataset "$DATASET" \
        --baseline "$B" \
        --model "$M" \
        --limit "$LIMIT" \
        --out-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1

      mark_done "$KEY"
    done
  done
  log "--> Phase 1 Complete."
fi

# ── Phase 2: Component Ablations ─────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "ablations" ]]; then
  ABLATIONS=("auditor_only_refusal" "sft_only" "no_tournament" "random_knob")
  MODELS=("SD 1.5")

  log "--> Starting Phase 2: Component Ablations"
  for A in "${ABLATIONS[@]}"; do
    for M in "${MODELS[@]}"; do
      KEY="ablation::${A}::${M}"
      if is_done "$KEY"; then
        log "  [skip] Ablation $A on $M already completed — skipping."
        continue
      fi

      log "  [Run] Executing Ablation: $A on Model: $M"
      python "$SCRIPT_DIR/ablations/run_ablations.py" \
        --dataset "$DATASET" \
        --config "$A" \
        --model "$M" \
        --limit "$LIMIT" \
        --out-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1

      mark_done "$KEY"
    done
  done
  log "--> Phase 2 Complete."
fi

# ── Phase 3: Auditor External Validity ───────────────────────────────────────
if [[ "$MODE" == "all" ]]; then
  KEY="auditor::external_validity"
  if ! is_done "$KEY"; then
    log "--> Starting Phase 3: Auditor External Validity Evaluation"
    python "$SCRIPT_DIR/evaluation/eval_auditor_external.py" \
      --out-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1
    mark_done "$KEY"
    log "--> Phase 3 Complete."
  else
    log "--> Phase 3: Auditor External Validity already evaluated — skipping."
  fi
fi

# ── Phase 4: Loss-Weight Sensitivity Sweep ───────────────────────────────────
if [[ "$MODE" == "all" ]]; then
  KEY="auditor::sensitivity_sweep"
  if ! is_done "$KEY"; then
    log "--> Starting Phase 4: Multi-Task Loss Weight Sensitivity Sweep"
    python "$SCRIPT_DIR/ablations/sensitivity_sweep.py" \
      --out-dir "$SCRIPT_DIR/results" >> "$LOG_FILE" 2>&1
    mark_done "$KEY"
    log "--> Phase 4 Complete."
  else
    log "--> Phase 4: Sensitivity Sweep already evaluated — skipping."
  fi
fi

# ── Phase 5: Generate Final Tables & Publication Figures ─────────────────────
log "--> Generating All Submission Tables and Figures"
python "$SCRIPT_DIR/tables/generate_all_artifacts.py" \
  --out-dir "$SCRIPT_DIR/tables/output" >> "$LOG_FILE" 2>&1

log "======================================================================"
log "  ALL RUNS & ARTIFACT GENERATION COMPLETE!"
log "  Tables and Figures written to: $SCRIPT_DIR/tables/output"
log "======================================================================"
