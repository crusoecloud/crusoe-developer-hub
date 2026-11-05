# Using Crusoe with OpenCode

Use Crusoe Managed Inference in [OpenCode](https://opencode.ai), the open-source terminal coding agent. OpenCode loads its provider list from [models.dev](https://models.dev), where Crusoe is listed, so no custom provider config is needed. A section in the OpenCode providers guide is in review ([anomalyco/opencode#55877](https://github.com/anomalyco/opencode/pull/55877)).

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- OpenCode installed (`curl -fsSL https://opencode.ai/install | bash`, or `npm i -g opencode-ai`)

## Setup

1. Run `opencode`, then `/connect` and search for **Crusoe**.
2. Paste your Crusoe API key when prompted.
3. Run `/models` and select a model, for example _GLM 5.3_ or _Kimi K2.6_.

Or set the key in the environment and pick the model in `opencode.json`:

```bash
export CRUSOE_API_KEY="your-api-key"
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "crusoe/zai/GLM-5.3"
}
```

## Notes

- Any Crusoe model in models.dev is selectable as `crusoe/<model id>`. Use `/models` to browse the current list.
- To point OpenCode at a Self-Serve Deployment instead of Serverless Inference, add a provider override with `options.baseURL` set to your deployment's URL, following the OpenCode docs on custom providers.

[Back to integrations](../README.md)
