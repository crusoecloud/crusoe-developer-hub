#!/usr/bin/env bash
set -euo pipefail

GATEWAY=${GATEWAY:-http://localhost:4000}
KEY=${LITELLM_MASTER_KEY:-sk-local-dev-master}

echo "== basic completion =="
curl -s "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Explain KV caching in two sentences."}]
  }' | python3 -m json.tool

echo "== streaming =="
curl -sN "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
