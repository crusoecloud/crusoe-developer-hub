#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "serve" ]; then
  vllm serve Qwen/Qwen3.8-27B \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:?set TOOL_CALL_PARSER to the parser for the Qwen3.8 family}" \
    --enable-prefix-caching --max-model-len 65536 --port 8000
  exit 0
fi

export LLM_BASE_URL="http://${VM_IP:?set VM_IP to the GPU VM address}:8000/v1"
export MODEL="Qwen/Qwen3.8-27B"
cd "$(dirname "$0")/.."
uv run --env-file .env python 03-agent/agent.py "${1:-MemoryAlloy KV cache}"
