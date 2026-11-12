#!/usr/bin/env bash
set -euo pipefail

curl "${LLM_BASE_URL:-https://api.inference.crusoecloud.com/v1}/chat/completions" \
  -H "Authorization: Bearer $CRUSOE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL:-zai-org/GLM-5.3}"'",
    "messages": [{"role": "user", "content": "What is 17 * 23? Use the tool."}],
    "tools": [{"type": "function", "function": {
      "name": "calc",
      "description": "Evaluate an arithmetic expression. Use it for all math.",
      "parameters": {"type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]}}}]
  }'
echo
