# Using Crusoe with opencodex

[opencodex](https://github.com/lidge-jun/opencodex) (`ocx`) is a provider proxy that lets Codex CLI, Claude Code, Claude Desktop, and other agent clients run on any model endpoint. Crusoe ships as a built-in preset since v2.59.0 ([lidge-jun/opencodex#4988](https://github.com/lidge-jun/opencodex/pull/4988), carrying [#4309](https://github.com/lidge-jun/opencodex/pull/4309)), so one command points those tools at Crusoe Serverless Inference.

## Prerequisites

- Node.js 20 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)

## Setup

```bash
npm i -g opencodex
ocx provider add crusoe      # prompts for the API key and discovers the live model list
```

Then select Crusoe as the active provider and launch your agent client through the proxy, following the [opencodex README](https://github.com/lidge-jun/opencodex#readme) for your client (Codex CLI, Claude Code, Claude Desktop, or Grok Build). The preset handles the endpoint, key validation, and model discovery.

## Notes

- The preset discovers models from `https://api.inference.crusoecloud.com/v1/models`, filters to public text and multimodal rows, and validates the key with a 401 check. Reasoning-effort ladders are exposed only for `openai/gpt-oss-120b`; GLM, Kimi, and Nemotron Omni toggle reasoning instead.
- Crusoe returns reasoning in the `reasoning` field of the message, interleaved with tool calls, and the preset maps it for the clients that render thinking.
- The preset lists Crusoe in opencodex's free-directory with the signup-credit note; see the [Crusoe pricing page](https://crusoe.ai/cloud/pricing) for per-model rates.

[Back to integrations](../README.md)
