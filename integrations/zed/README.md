# Using Crusoe with Zed

Use Crusoe Managed Inference models in [Zed](https://zed.dev)'s AI features (Zed Agent, Inline Assistant, commit message generation) through Zed's OpenAI-compatible provider support.

## Prerequisites

- A Crusoe account and an Inference API key, created in the [Crusoe Console](https://console.crusoecloud.com/) under **Security > Inference API Key**
- Zed installed ([zed.dev](https://zed.dev))

## Setup via the UI

1. Open Zed's Agent Settings (`agent: open settings`).
2. In the **LLM Providers** section, click **Add Provider**.
3. Fill in the fields:
   - **Provider name:** `Crusoe`
   - **API URL:** `https://api.inference.crusoecloud.com/v1`
   - **Model ID:** `zai/GLM-5.2`
   - **Context window:** `256000`
4. Enter your Crusoe API key when prompted. Zed stores it in the system keychain.

## Setup via settings.json

Alternatively, configure the provider directly in your Zed `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "crusoe": {
        "api_url": "https://api.inference.crusoecloud.com/v1",
        "available_models": [
          {
            "name": "zai/GLM-5.2",
            "display_name": "GLM 5.2 (Crusoe)",
            "max_tokens": 256000,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true
            }
          },
          {
            "name": "openai/gpt-oss-120b",
            "display_name": "gpt-oss-120b (Crusoe)",
            "max_tokens": 131072,
            "reasoning_effort": "medium",
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true
            }
          },
          {
            "name": "meta-llama/Llama-3.3-70B-Instruct",
            "display_name": "Llama 3.3 70B (Crusoe)",
            "max_tokens": 131072,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": false
            }
          }
        ]
      }
    }
  }
}
```

With the provider ID `crusoe`, Zed also reads the API key from the `CRUSOE_API_KEY` environment variable (Zed generates the variable name from the provider ID). A non-empty environment variable takes precedence over a key saved in the keychain:

```bash
export CRUSOE_API_KEY="your-api-key"
```

## Notes

- **Reasoning models:** GLM 5.2 returns reasoning interleaved with tool calls. `openai/gpt-oss-120b` supports `reasoning_effort` values of `low`, `medium`, and `high`.
- **Model catalog:** The current model list is available from `https://api.inference.crusoecloud.com/v1/models`. Add any model from the catalog as an entry in `available_models`.
- **Pricing:** See [Crusoe pricing](https://crusoe.ai/cloud/pricing) for per-model rates.
