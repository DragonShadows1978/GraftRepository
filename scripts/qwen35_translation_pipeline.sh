#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt/ForgeRealm/GraftRepository}
POC=${POC:-/mnt/ForgeRealm/qwen35_graft_translation_poc}
Q35_2B=${Q35_2B:-/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc}
Q35_9B=${Q35_9B:-/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
TCPATH=${TCPATH:-/mnt/ForgeRealm/Project-Tensor/tensor_cuda}
LOCK=${LOCK:-/tmp/qwen35_translation_pipeline.lock}
PYTHON_BIN=${PYTHON_BIN:-python3}

SOURCE_MAX_CHUNKS=${SOURCE_MAX_CHUNKS:-64}
TARGET_MAX_CHUNKS=${TARGET_MAX_CHUNKS:-16}
RIDGE_LAMBDA=${RIDGE_LAMBDA:-1e-4}
TOPK=${TOPK:-16}
BINDING_MAX_PROBES=${BINDING_MAX_PROBES:-32}

export PYTHONPATH="${REPO}:${TCPATH}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: scripts/qwen35_translation_pipeline.sh <command>

Commands:
  next       Run one locked pipeline-next stage.
  status     Refresh non-GPU pipeline_status.json and print it.
  history    Print the last 20 pipeline_history.jsonl rows.
  paths      Print resolved paths and batch knobs.

Environment overrides:
  REPO POC Q35_2B Q35_9B TCPATH LOCK PYTHON_BIN
  SOURCE_MAX_CHUNKS TARGET_MAX_CHUNKS RIDGE_LAMBDA TOPK BINDING_MAX_PROBES
EOF
}

pipeline_next() {
  cd "$REPO"
  flock -n "$LOCK" "$PYTHON_BIN" scripts/qwen35_graft_translate_poc.py pipeline-next \
    --root "$POC" \
    --source-model-dir "$Q35_2B" \
    --target-model-dir "$Q35_9B" \
    --layers all \
    --source-max-chunks "$SOURCE_MAX_CHUNKS" \
    --target-max-chunks "$TARGET_MAX_CHUNKS" \
    --ridge-lambda "$RIDGE_LAMBDA" \
    --topk "$TOPK" \
    --binding-max-probes "$BINDING_MAX_PROBES"
}

pipeline_status() {
  cd "$REPO"
  exec "$PYTHON_BIN" scripts/qwen35_graft_translate_poc.py pipeline-status \
    --root "$POC" \
    --write-status
}

print_paths() {
  cat <<EOF
REPO=$REPO
POC=$POC
Q35_2B=$Q35_2B
Q35_9B=$Q35_9B
TCPATH=$TCPATH
LOCK=$LOCK
SOURCE_MAX_CHUNKS=$SOURCE_MAX_CHUNKS
TARGET_MAX_CHUNKS=$TARGET_MAX_CHUNKS
RIDGE_LAMBDA=$RIDGE_LAMBDA
TOPK=$TOPK
BINDING_MAX_PROBES=$BINDING_MAX_PROBES
EOF
}

cmd=${1:-}
case "$cmd" in
  next)
    pipeline_next
    ;;
  status)
    pipeline_status
    ;;
  history)
    if [[ -f "$POC/pipeline_history.jsonl" ]]; then
      tail -n 20 "$POC/pipeline_history.jsonl"
    else
      echo "No pipeline history yet: $POC/pipeline_history.jsonl"
    fi
    ;;
  paths)
    print_paths
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
