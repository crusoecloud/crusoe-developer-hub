# Using Crusoe with Continue

Use Crusoe Managed Inference models for chat, edit, and apply in [Continue](https://continue.dev), the open-source AI code assistant for VS Code and JetBrains. Continue's OpenAI-compatible provider works with Crusoe today; a native `crusoe` provider with a docs page is in review upstream ([continuedev/continue#13136](https://github.com/continuedev/continue/pull/13136)) and shortens the config to `provider: crusoe`.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- Continue installed in your editor

## Setup

Add Crusoe models to your Continue `config.yaml` (open it from the Continue sidebar's settings). The `openai` provider plus `apiBase` is the OpenAI-compatible path:

```yaml
name: Crusoe
version: 0.0.1
schema: v1
models:
  - name: GLM 5.3 (Crusoe)
    provider: openai
    model: zai/GLM-5.3
    apiBase: https://api.inference.crusoecloud.com/v1
    apiKey: ${{ secrets.CRUSOE_API_KEY }}
    roles: [chat, edit, apply]
    capabilities: [tool_use]
  - name: gpt-oss-120b (Crusoe)
    provider: openai
    model: openai/gpt-oss-120b
    apiBase: https://api.inference.crusoecloud.com/v1
    apiKey: ${{ secrets.CRUSOE_API_KEY }}
    roles: [chat]
    defaultCompletionOptions:
      reasoningEffort: medium
```

Store the key once in Continue's secrets rather than in the file, or replace the reference with the literal key in a local-only config.

After continuedev/continue#13136 ships, the same entries become:

```yaml
  - name: GLM 5.3 (Crusoe)
    provider: crusoe
    model: zai/GLM-5.3
    apiKey: ${{ secrets.CRUSOE_API_KEY }}
```

## Notes

- Models with tool calling (GLM, Kimi, gpt-oss, Llama 3.3) power Continue's Agent mode; add `capabilities: [tool_use]` so Continue offers it.
- For autocomplete, prefer a small fast model and the `autocomplete` role; check the current catalog at `https://api.inference.crusoecloud.com/v1/models` for FIM-capable models before assigning the role.
- The legacy `config.json` format takes the same fields: `"provider": "openai"`, `"model"`, `"apiBase"`, `"apiKey"`.

[Back to integrations](../README.md)
