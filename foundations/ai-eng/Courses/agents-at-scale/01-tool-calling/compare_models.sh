#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
QUESTION="${1:-What is 17 * 23, and then that times 3?}"
for model in "Qwen/Qwen3.8-27B" "zai-org/GLM-5.3"; do
  echo "== MODEL=$model =="
  MODEL="$model" python loop.py "$QUESTION"
  echo
done
