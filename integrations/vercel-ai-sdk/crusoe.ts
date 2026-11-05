// Shared Crusoe provider instance for the recipes. Reads CRUSOE_API_KEY from the environment.
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

const apiKey = process.env.CRUSOE_API_KEY;
if (!apiKey) {
  throw new Error("Set CRUSOE_API_KEY to your Crusoe Inference API key.");
}

export const crusoe = createOpenAICompatible({
  name: "crusoe",
  baseURL: "https://api.inference.crusoecloud.com/v1",
  apiKey,
});

export const DEFAULT_MODEL = "zai/GLM-5.3";
