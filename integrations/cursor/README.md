# Using Crusoe with Cursor

Use Crusoe Managed Inference models in [Cursor](https://cursor.com) by overriding the OpenAI base URL with Crusoe's OpenAI-compatible endpoint.

## Prerequisites

- A Crusoe account and an Inference API key, created in the [Crusoe Console](https://console.crusoe.ai/) under **Security > Inference API Key**
- Cursor installed ([cursor.com](https://cursor.com))

## Setup

1. Open Cursor Settings (`Cmd/Ctrl + Shift + J`) and go to the **Models** tab.
2. Scroll to **API Keys** and expand the **OpenAI API Key** section.
3. Enable **Override OpenAI Base URL** and set it to:

   ```
   https://api.inference.crusoecloud.com/v1
   ```

4. Paste your Crusoe API key in the OpenAI API Key field and click **Verify**.
5. Under the model list, click **Add model** and add a Crusoe model ID, for example:

   ```
   zai/GLM-5.2
   ```

6. Enable the custom model and select it from the model picker in chat.

## Recommended models

| Model | Context window | Best for |
|---|---|---|
| `zai/GLM-5.2` | 256,000 | Coding and agentic tool use |
| `openai/gpt-oss-120b` | 131,072 | Reasoning with effort control |
| `moonshotai/Kimi-K2.6` | 262,144 | Agentic workflows |
| `meta-llama/Llama-3.3-70B-Instruct` | 131,072 | General chat |

## Limitations

- Cursor's base-URL override replaces OpenAI models for the session: custom API keys with an overridden base URL disable Cursor's built-in models while active. Toggle the override off to switch back.
- Custom models cover chat features. Cursor-specific features such as Tab autocomplete run on Cursor's own infrastructure and are not affected by this setting.
- The model ID must match the Crusoe catalog exactly, including the org prefix (for example `zai/GLM-5.2`, not `GLM-5.2`). The current catalog is available from `https://api.inference.crusoecloud.com/v1/models`.
