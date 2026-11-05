# Using Crusoe with Kilo Code

Use Crusoe Managed Inference in [Kilo Code](https://kilocode.ai), the open-source AI coding agent for VS Code, JetBrains, and the CLI. Crusoe is a built-in provider: Kilo's catalog is generated from [models.dev](https://models.dev), where Crusoe has been listed since [anomalyco/models.dev#3769](https://github.com/anomalyco/models.dev/pull/3769). A dedicated docs page is in review ([Kilo-Org/kilocode#14267](https://github.com/Kilo-Org/kilocode/pull/14267)).

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)
- Kilo Code installed in your editor, or the `kilo` CLI

## Setup in VS Code or JetBrains

1. Open **Settings** (gear icon) and go to the **Providers** tab.
2. Click **Connect provider**, search for **Crusoe**, and select it.
3. Enter your Crusoe API key.
4. Pick a model. Kilo fetches the available models automatically.

## Setup in the CLI

Run `kilo`, then connect interactively:

```bash
kilo
# inside Kilo:
/connect        # select Crusoe, paste the key
/models         # pick a model, for example zai/GLM-5.3
```

Or configure it in `~/.config/kilo/kilo.json` (or a project-level `./kilo.json`) and keep the key in the environment:

```bash
export CRUSOE_API_KEY="your-api-key"
```

```jsonc
{
  "provider": {
    "crusoe": {
      "env": ["CRUSOE_API_KEY"]
    }
  },
  "model": "crusoe/zai/GLM-5.3"
}
```

## Notes

- Model ids follow the catalog exactly, for example `crusoe/zai/GLM-5.3`, `crusoe/moonshotai/Kimi-K2.6`, `crusoe/openai/gpt-oss-120b`.
- Context windows and pricing come from models.dev, so Kilo's context management and cost display are accurate for Crusoe models.
- Rate limits for Serverless Inference are published in the [Crusoe docs](https://docs.crusoecloud.com/serverless-inference/overview); long agent sessions on a single key can hit them, so a Self-Serve Deployment is the upgrade path for a team.

[Back to integrations](../README.md)
