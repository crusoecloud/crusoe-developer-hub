import type { ProviderCatalogResult } from "openclaw/plugin-sdk/plugin-entry";

type CrusoeProviderConfig = Extract<
  NonNullable<ProviderCatalogResult>,
  { provider: unknown }
>["provider"];

export type CrusoeModel = CrusoeProviderConfig["models"][number];

export const CRUSOE_PROVIDER_ID = "crusoe";
export const CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1";
export const CRUSOE_DEFAULT_MODEL_ID = "openai/gpt-oss-120b";
export const CRUSOE_DEFAULT_MODEL_REF = `${CRUSOE_PROVIDER_ID}/${CRUSOE_DEFAULT_MODEL_ID}`;
export const CRUSOE_DEFAULT_CONTEXT_WINDOW = 131072;
export const CRUSOE_DEFAULT_MAX_TOKENS = 32768;

const SHARED_COMPAT = {
  supportsUsageInStreaming: true,
  supportsPromptCacheKey: true,
  supportsStrictMode: false,
} as const;

const TOGGLE_LEVEL_MAP = { minimal: null, medium: null, high: null, xhigh: null, max: null };

function toggleCompat(defaultThinkingLevel: "off" | "low") {
  return {
    ...SHARED_COMPAT,
    supportsReasoningEffort: true,
    supportedReasoningEfforts: ["none", "low"],
    reasoningEffortMap: { off: "none", low: "low" },
    thinkingLevelLabels: { low: "on" },
    defaultThinkingLevel,
  };
}

export const CRUSOE_STATIC_MODELS: readonly CrusoeModel[] = [
  {
    id: "openai/gpt-oss-120b",
    name: "GPT-OSS 120B",
    reasoning: true,
    thinkingLevelMap: { off: null, minimal: null, xhigh: null, max: null },
    input: ["text"],
    contextWindow: 131072,
    maxTokens: 32768,
    cost: { input: 0.05, output: 0.25, cacheRead: 0.05, cacheWrite: 0 },
    compat: {
      ...SHARED_COMPAT,
      supportsReasoningEffort: true,
      supportedReasoningEfforts: ["low", "medium", "high"],
      reasoningEffortMap: { low: "low", medium: "medium", high: "high" },
      defaultThinkingLevel: "medium",
    },
  },
  {
    id: "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
    name: "Nemotron 3 Nano Omni 30B A3B Reasoning",
    reasoning: true,
    thinkingLevelMap: TOGGLE_LEVEL_MAP,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 65536,
    cost: { input: 0.3, output: 1.83, cacheRead: 0.3, cacheWrite: 0 },
    compat: toggleCompat("low"),
  },
  {
    id: "google/gemma-4-31b-it",
    name: "Gemma 4 31B IT",
    reasoning: true,
    thinkingLevelMap: TOGGLE_LEVEL_MAP,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 32768,
    cost: { input: 0.14, output: 0.4, cacheRead: 0.14, cacheWrite: 0 },
    compat: toggleCompat("off"),
  },
];

export function cloneCrusoeStaticModels(): CrusoeModel[] {
  return CRUSOE_STATIC_MODELS.map((model) => structuredClone(model));
}

export function findCrusoeStaticModel(modelId: string): CrusoeModel | undefined {
  return CRUSOE_STATIC_MODELS.find((model) => model.id === modelId);
}
