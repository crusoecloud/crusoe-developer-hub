# Using Crusoe with Cline

Use Crusoe Managed Inference models in [Cline](https://cline.bot), the open-source coding agent for VS Code and the terminal. Crusoe is a built-in Cline provider ([cline/cline#12267](https://github.com/cline/cline/pull/12267)), and its model list comes from the [models.dev](https://models.dev) catalog, so new Crusoe models appear in Cline without an update on your side.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- Cline installed from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev), or the Cline CLI

## Setup in VS Code

1. Open Cline and click the settings gear.
2. Under **API Provider**, choose **Crusoe**.
3. Paste your Crusoe API key.
4. Pick a model. `zai/GLM-5.3` is the default; `moonshotai/Kimi-K2.6` and `deepseek-ai/DeepSeek-V4-Pro` are strong alternatives for agentic coding.

## Setup in the CLI

Cline reads the key from the environment when the provider is `crusoe`:

```bash
export CRUSOE_API_KEY="your-api-key"
cline auth   # choose Crusoe, confirm the key, pick a model
```

## Notes

- Crusoe is an OpenAI-compatible provider in Cline, so Plan and Act modes, MCP servers, and checkpoints all work as with any other provider.
- Models that interleave reasoning with tool calls (GLM, Kimi) give Cline better multi-step edits; pick a Flash or Nano model for cheaper, faster passes.
- Pricing per model is on the [Crusoe pricing page](https://crusoe.ai/cloud/pricing); Cline shows the cost of each task from the usage it receives.

[Back to integrations](../README.md)
