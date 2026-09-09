#!/usr/bin/env bash
set -euo pipefail

GATEWAY=${GATEWAY:-http://localhost:4000}
KEY=${LITELLM_MASTER_KEY:-sk-local-dev-master}

echo "== liveliness =="
curl -sf "$GATEWAY/health/liveliness" && echo

echo "== models =="
curl -sf -H "Authorization: Bearer $KEY" "$GATEWAY/v1/models" \
  | python3 -c "import json,sys; [print(' -', m['id']) for m in json.load(sys.stdin)['data']]"

echo "== chat completion (glm-5.2) =="
curl -sf -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "glm-5.2", "messages": [{"role": "user", "content": "Reply with exactly: gateway ok"}], "max_tokens": 200}' \
  "$GATEWAY/v1/chat/completions" \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print('model:', r['model']); print('reply:', r['choices'][0]['message']['content'])"

echo "== smoke passed =="
