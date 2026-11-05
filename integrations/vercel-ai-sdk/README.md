# Vercel AI SDK with Crusoe

Use [Crusoe Managed Inference](https://www.crusoe.ai/cloud/managed-inference) from the [Vercel AI SDK](https://ai-sdk.dev/) in Node.js, Next.js, or any TypeScript app. Crusoe is listed as an AI SDK community provider ([vercel/ai#18466](https://github.com/vercel/ai/pull/18466)). The recipes here use `@ai-sdk/openai-compatible`, which tracks the current AI SDK major and needs only the Crusoe base URL and key.

## Prerequisites

- Node.js 20 or later
- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)

## Setup

```bash
cd integrations/vercel-ai-sdk
npm install
export CRUSOE_API_KEY="your-api-key"
```

## Quick start

```ts
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText } from "ai";

const crusoe = createOpenAICompatible({
  name: "crusoe",
  baseURL: "https://api.inference.crusoecloud.com/v1",
  apiKey: process.env.CRUSOE_API_KEY,
});

const { text } = await generateText({
  model: crusoe("zai/GLM-5.3"),
  prompt: "Write a haiku about sustainable computing.",
});
console.log(text);
```

## Recipes

| Script | What it shows |
| --- | --- |
| [`generate.ts`](generate.ts) | `generateText` with a system prompt and token usage |
| [`stream.ts`](stream.ts) | `streamText` written to stdout as tokens arrive |
| [`tools.ts`](tools.ts) | A Zod-typed tool and a multi-step run with `stopWhen` |
| [`structured.ts`](structured.ts) | `generateObject` returning a typed object validated against a Zod schema |

```bash
npm run generate
npm run stream
npm run tools
npm run structured
```

Each script makes live requests that incur inference charges.

## Using `crusoe-ai-provider`

The community package [`crusoe-ai-provider`](https://www.npmjs.com/package/crusoe-ai-provider) wraps the same endpoint with a default instance and reads `CRUSOE_API_KEY` for you. It targets AI SDK 5, so use it in projects on that major:

```ts
import { crusoe } from "crusoe-ai-provider";
import { generateText } from "ai";

const { text } = await generateText({ model: crusoe("zai/GLM-5.3"), prompt: "Hello" });
```

## Notes

- Any id from `https://api.inference.crusoecloud.com/v1/models` works. `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K2.6`, and `openai/gpt-oss-120b` are good streaming and tool-use picks.
- The provider instance works anywhere the AI SDK takes a model: `useChat` route handlers, `generateObject`, agents, and `ai/rsc`.
- Pass reasoning effort for gpt-oss through `providerOptions: { crusoe: { reasoningEffort: "medium" } }`.

[Back to integrations](../README.md)
