# Crusoe Managed Inference Postman Collection

A Postman collection covering the Crusoe Managed Inference API (OpenAI-compatible).

## Requests included

1. **List models** - the current catalog from `/v1/models`
2. **Chat completion** - basic request with `zai/GLM-5.2`
3. **Chat completion (streaming)** - SSE streaming
4. **Tool calling** - function-calling request and `tool_calls` response shape
5. **Reasoning effort** - `reasoning_effort` on `openai/gpt-oss-120b` (low/medium/high) with notes on toggle-style models
6. **Structured output** - JSON mode with `google/gemma-4-31b-it`

## Setup

1. Import [crusoe-managed-inference.postman_collection.json](crusoe-managed-inference.postman_collection.json) into Postman (File > Import).
2. Create an Inference API key in the [Crusoe Console](https://console.crusoecloud.com/) under **Security > Inference API Key**.
3. Set the `CRUSOE_API_KEY` collection variable (Collection > Variables). Auth is bearer-token at the collection level, so every request inherits it.

## Publishing options

- **Postman Public API Network:** publish from a Crusoe team workspace (Workspace > Share > Public) so the collection is discoverable at postman.com/explore. This is how providers typically appear in Postman's network and mirrors what other inference providers do.
- **Run in Postman button:** once public, Postman generates an embeddable button for docs.crusoecloud.com.
- **Direct file distribution:** link the JSON from the docs for manual import.
