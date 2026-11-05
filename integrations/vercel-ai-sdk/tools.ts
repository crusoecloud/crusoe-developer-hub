// A Zod-typed tool and a multi-step run on Crusoe. Incurs inference charges.
import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";

import { crusoe, DEFAULT_MODEL } from "./crusoe.js";

const rates: Record<string, { input: number; output: number }> = {
  "zai/GLM-5.3": { input: 0.4, output: 1.26 },
  "openai/gpt-oss-120b": { input: 0.1, output: 0.5 },
  "meta-llama/Llama-3.3-70B-Instruct": { input: 0.25, output: 0.75 },
};

const getRate = tool({
  description: "Price per million input and output tokens for a model on Crusoe.",
  inputSchema: z.object({
    modelId: z.string().describe('Model id as listed on Crusoe, e.g. "zai/GLM-5.3"'),
  }),
  execute: async ({ modelId }) =>
    rates[modelId] ?? { error: `unknown model ${modelId}`, known: Object.keys(rates) },
});

const { text, steps } = await generateText({
  model: crusoe(DEFAULT_MODEL),
  system: "You estimate inference spend. Use getRate for every price; never guess.",
  prompt:
    "A support bot sends 2M input and 400k output tokens a day on gpt-oss-120b. " +
    "What does a 30-day month cost, and how much more would GLM-5.3 be?",
  tools: { getRate },
  stopWhen: stepCountIs(5),
});

console.log(text);
console.log(
  "\n[tool calls]",
  steps.flatMap((s) => s.toolCalls.map((c) => `${c.toolName}(${JSON.stringify(c.input)})`)),
);
