# openclaw-plugin-crusoe

An [OpenClaw](https://openclaw.ai/) provider plugin for [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference). It adds Crusoe to OpenClaw's model setup, so an OpenClaw assistant can run on the open-weight models Crusoe serves behind `https://api.inference.crusoecloud.com/v1`.

OpenClaw routes third-party providers through plugins that the vendor publishes and maintains. This folder is the source of that plugin.

| Property        | Value                                      |
| --------------- | ------------------------------------------ |
| Provider id     | `crusoe`                                   |
| Auth env var    | `CRUSOE_API_KEY`                           |
| Onboarding flag | `--auth-choice crusoe-api-key`             |
| Direct CLI flag | `--crusoe-api-key <key>`                   |
| API             | OpenAI-compatible (`openai-completions`)   |
| Base URL        | `https://api.inference.crusoecloud.com/v1` |
| Default model   | `crusoe/openai/gpt-oss-120b`               |

## Prerequisites

- OpenClaw 2026.11.3 or later
- Node.js 24.16 or later on the 24 line, or 26.1 or later, to build the plugin
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Intelligence Foundry > Inference > Create API Key**)

## Install

Build the plugin from this folder and install the archive:

```bash
cd integrations/openclaw
npm install
npm run build
npm pack
openclaw plugins install ./openclaw-plugin-crusoe-0.1.0.tgz --force --accept-capabilities
```

`--force` confirms an install from a local archive, and `--accept-capabilities` accepts the provider capability the plugin declares.

Once the package is published on [ClawHub](https://clawhub.ai/), the install is one command:

```bash
openclaw plugins install openclaw-plugin-crusoe
```

## Setup

Give OpenClaw the key:

```bash
openclaw onboard --auth-choice crusoe-api-key
```

For scripted installs:

```bash
openclaw onboard --non-interactive --accept-risk --skip-health \
  --auth-choice crusoe-api-key \
  --crusoe-api-key "$CRUSOE_API_KEY"
```

Check that the models are available:

```bash
openclaw models list --provider crusoe --refresh
```

## Models

With a key saved, the plugin lists every public chat model the key can see from `GET /v1/models` and keeps Crusoe's ids, context windows, and prices. A new Crusoe model needs no plugin release: pick it as `crusoe/<model id>` once it appears in the list. Private dedicated deployments and non-chat rows such as embeddings are not listed as chat models.

These models carry reviewed metadata:

| Model ref                                              | Input       | Context | Thinking                         |
| ------------------------------------------------------ | ----------- | ------- | -------------------------------- |
| `crusoe/openai/gpt-oss-120b`                           | text        | 131,072 | low/medium/high (default medium) |
| `crusoe/nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` | text, image | 262,144 | off/on (default on)              |
| `crusoe/google/gemma-4-31b-it`                         | text, image | 262,144 | off/on (default off)             |

If the model list cannot be read, the plugin serves these three rows. Check the [available models page](https://docs.crusoecloud.com/serverless-inference/available-models) for the current catalog.

## Thinking

Crusoe models control reasoning through the OpenAI `reasoning_effort` field.

- **GPT-OSS 120B** accepts `low`, `medium`, and `high`, and each level changes how long the model reasons. It cannot turn reasoning off.
- **Nemotron 3 Nano Omni** and **Gemma 4** are hybrid-reasoning models: `none` turns reasoning off and any other level turns it on. The picker offers `off` and `on`.

Models that appear only in the live list are sent no `reasoning_effort` and reason the way the endpoint defaults them. Configure a custom model row if one needs a specific ladder.

## Custom base URL

Setting `models.providers.crusoe.baseUrl` routes chat requests to that URL. The plugin then serves the reviewed rows and does not call the model list, so the key is listed only against the Crusoe host.

## Develop

```bash
npm install
npm run build
npm test
npm run validate
```

`npm run validate` builds the plugin and runs `clawhub package validate . --out .clawhub-validation`, the same inspection ClawHub applies on publish.

The reviewed rows live in `src/models.ts` and are mirrored in `openclaw.plugin.json`. A test fails when they drift.

## Publish

Releases go to ClawHub from a maintainer's machine:

```bash
npm exec clawhub -- login
npm run validate
npm exec clawhub -- package publish . --dry-run
npm exec clawhub -- package publish . --wait
```

Bump `version` in `package.json` and `openclaw.plugin.json` together before each release.

## References

- [Crusoe Serverless Inference docs](https://docs.crusoecloud.com/serverless-inference/overview)
- [Building OpenClaw provider plugins](https://docs.openclaw.ai/plugins/sdk-provider-plugins)
- [Crusoe Console](https://console.crusoecloud.com/)
