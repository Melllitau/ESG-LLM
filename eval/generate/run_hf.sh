#!/usr/bin/env bash
#chmod +x run_hf.sh
set -Eeuo pipefail

# ── EDITABLE ───────────────────────────────────────────
MODELS=(
  "g-assismoraes/Qwen3-4B-esg-qa"
  "g-assismoraes/Qwen3-4B-inst-irm-esg-ep3"
  "g-assismoraes/Qwen3-4B-esg-ep3"

  "Qwen/Qwen3-4B-Instruct-2507"

)

#EVAL DATASET
INPUT_JSON=""   
RESULTS_ROOT="./results"
PYTHON_BIN="python3"
RUN_HF_PY="run_hf.py"

# ── Sampling parameters ─────────────────────────────────────────────
SAMPLING_MS=250              # intervalo de amostragem da VRAM (ms)
ENABLE_VRAM_SAMPLER=true     # false para pular amostragem

mkdir -p "$RESULTS_ROOT"
mkdir -p "$IR_DIR"

# ── Sampling parameters ─────────────────────────────────────────────
SAMPLING_MS=250              # intervalo de amostragem da VRAM (ms)
ENABLE_VRAM_SAMPLER=true     # false para pular amostragem

# ── Helpers ─────────────────────────────────────────────
all_gpu_mem_max_mib() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo ""; return; fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'm<$1{m=$1} END{if(m=="" ) print ""; else print m}'
}

prime_one_sample() {
  local f="$1"
  local cur="$(all_gpu_mem_max_mib || echo "")"
  [[ "$cur" =~ ^[0-9]+$ ]] && echo "$cur" >> "$f"
}

sample_until_pid_ends() {  # args: sample_file pid sampling_ms
  local f="$1" pid="$2" ms="$3"
  : > "$f"
  prime_one_sample "$f"
  while kill -0 "$pid" 2>/dev/null; do
    local cur="$(all_gpu_mem_max_mib || echo "")"
    [[ "$cur" =~ ^[0-9]+$ ]] && echo "$cur" >> "$f"
    sleep "$(awk "BEGIN{printf \"%.3f\", ${ms}/1000}")"
  done
}

max_from_samples() {  # sample_file → integer MiB
  local f="$1"
  [[ -s "$f" ]] || { echo "0"; return; }
  awk 'm<$1{m=$1} END{print (m==""?0:m+0)}' "$f"
}

to_gib() { $PYTHON_BIN - "$1" <<'PY'
import sys; m=float(sys.argv[1] or 0); print(f"{m/1024:.3f}")
PY
}

to_gb_dec() { $PYTHON_BIN - "$1" <<'PY'
import sys; m=float(sys.argv[1] or 0); print(f"{m*1048576/1_000_000_000:.3f}")
PY
}

write_max_if_greater() {
  local file="$1" new="$2"
  if ! [[ "$new" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "WARN: write_max_if_greater - invalid new value: '$new'" >&2
    return 1
  fi
  if [[ -f "$file" ]]; then
    local old
    old="$(cat "$file" 2>/dev/null | head -n1 | awk '{print $1}')"
    if ! [[ "$old" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      old=0
    fi
    if awk -v n="$new" -v o="$old" 'BEGIN{exit !(n>o)}'; then
      echo "$new" > "$file"
    fi
  else
    echo "$new" > "$file"
  fi
}

SUMMARY="${RESULTS_ROOT}/vram_summary.tsv"
[[ -f "$SUMMARY" ]] || echo -e "model\tllm_mib\tllm_gib\tllm_gb" > "$SUMMARY"

# ── Main loop ──────────────────────────────────────────


for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=$(echo "$MODEL" | sed -e 's#[/:]#_#g' -e 's#\.##g')
  MODEL_DIR="${RESULTS_ROOT}/${SAFE_MODEL}"
  LLM_DIR="${MODEL_DIR}/llm"
  AGENT_DIR="${MODEL_DIR}/agent"
  mkdir -p "$LLM_DIR" "$AGENT_DIR"

  # LLM-only outputs and VRAM go to llm/
  LLM_OUT_JSON="${LLM_DIR}/${SAFE_MODEL}_llm.json"
  LLM_VRAM_MIB="${LLM_DIR}/vram_max_mib.txt"
  LLM_VRAM_GIB="${LLM_DIR}/vram_max_gib.txt"
  LLM_VRAM_GB="${LLM_DIR}/vram_max_gb.txt"
  LLM_SAMPLES="$(mktemp -t vram_llm_${SAFE_MODEL}.XXXXXX)"

  # Agent outputs and VRAM go to agent/ (comentado)
  AGENT_OUT_JSON="${AGENT_DIR}/${SAFE_MODEL}_agent.json"
  AGENT_VRAM_MIB="${AGENT_DIR}/vram_max_mib.txt"
  AGENT_VRAM_GIB="${AGENT_DIR}/vram_max_gib.txt"
  AGENT_VRAM_GB="${AGENT_DIR}/vram_max_gb.txt"

  echo "────────────────────────────────────────────────────────────"
  echo "Model: $MODEL"
  echo "Input: $INPUT_JSON | IR: $IR_DIR | Results: $RESULTS_ROOT"
  echo "────────────────────────────────────────────────────────────"

  # # ── LLM-only mode ────────────────────────────────────
  echo "[LLM] Running $MODEL ..."
  LLM_LOG="${LLM_DIR}/run.log"
  "$PYTHON_BIN" "$RUN_HF_PY" \
    --mode llm \
    --model "$MODEL" \
    --input "$INPUT_JSON" \
    --output "$LLM_OUT_JSON" \
    > "$LLM_LOG" 2>&1 &
  LLM_PID=$!
  if $ENABLE_VRAM_SAMPLER && command -v nvidia-smi >/dev/null 2>&1; then
    sample_until_pid_ends "$LLM_SAMPLES" "$LLM_PID" "$SAMPLING_MS"
  fi
  wait "$LLM_PID"; LLM_RC=$?
  LLM_MAX_MiB="$(max_from_samples "$LLM_SAMPLES" || echo 0)"
  write_max_if_greater "$LLM_VRAM_MIB" "$LLM_MAX_MiB"
  write_max_if_greater "$LLM_VRAM_GIB" "$(to_gib "$LLM_MAX_MiB")"
  write_max_if_greater "$LLM_VRAM_GB" "$(to_gb_dec "$LLM_MAX_MiB")"
  rm -f "$LLM_SAMPLES"
  echo "[LLM] Exit code: $LLM_RC | Max: ${LLM_MAX_MiB} MiB ($(to_gib "$LLM_MAX_MiB") GiB, $(to_gb_dec "$LLM_MAX_MiB") GB)"
  [[ $LLM_RC -ne 0 ]] && echo "[LLM] WARNING: non-zero exit; see $LLM_OUT_JSON."
  echo -e "${MODEL}\t${LLM_MAX_MiB}\t$(to_gib "$LLM_MAX_MiB")\t$(to_gb_dec "$LLM_MAX_MiB")" >> "$SUMMARY"

  # ── Agent mode (com IR/RAG) ──────────────────────────
  AGENT_LOG="${AGENT_DIR}/run.log"
  echo "[AGENT] Running $MODEL ... (log: $AGENT_LOG)"
  "$PYTHON_BIN" "$RUN_HF_PY" \
    --mode agent \
    --model "$MODEL" \
    --input "$INPUT_JSON" \
    --use_ir \
    --embed_model "$EMBED_MODEL" \
    --persist_dir "$IR_DIR" \
    --top_k "$TOP_K" \
    --output "$AGENT_OUT_JSON" \
    > "$AGENT_LOG" 2>&1 &
  AGENT_PID=$!
  AGENT_SAMPLES="$(mktemp -t vram_agent_${SAFE_MODEL}.XXXXXX)"
  if $ENABLE_VRAM_SAMPLER && command -v nvidia-smi >/dev/null 2>&1; then
    sample_until_pid_ends "$AGENT_SAMPLES" "$AGENT_PID" "$SAMPLING_MS"
  fi
  wait "$AGENT_PID"; AGENT_RC=$?
  AGENT_MAX_MiB="$(max_from_samples "$AGENT_SAMPLES" || echo 0)"
  write_max_if_greater "$AGENT_VRAM_MIB" "$AGENT_MAX_MiB"
  write_max_if_greater "$AGENT_VRAM_GIB" "$(to_gib "$AGENT_MAX_MiB")"
  write_max_if_greater "$AGENT_VRAM_GB" "$(to_gb_dec "$AGENT_MAX_MiB")"
  rm -f "$AGENT_SAMPLES"
  echo "[AGENT] Exit code: $AGENT_RC | Max: ${AGENT_MAX_MiB} MiB ($(to_gib "$AGENT_MAX_MiB") GiB, $(to_gb_dec "$AGENT_MAX_MiB") GB)"
  [[ $AGENT_RC -ne 0 ]] && echo "[AGENT] WARNING: non-zero exit; see $AGENT_OUT_JSON."
  echo -e "${MODEL}\t${AGENT_MAX_MiB}\t$(to_gib "$AGENT_MAX_MiB")\t$(to_gb_dec "$AGENT_MAX_MiB")" >> "$SUMMARY"
done

echo "Done. Results at: ${RESULTS_ROOT}"
echo "VRAM summary: ${SUMMARY}"
