import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { createProviderApiKeyAuthMethod } from "openclaw/plugin-sdk/provider-auth";
import { fetchCrusoeModelRows, projectCrusoeModels } from "./live-models.js";
import {
  cloneCrusoeStaticModels,
  CRUSOE_BASE_URL,
  CRUSOE_DEFAULT_CONTEXT_WINDOW,
  CRUSOE_DEFAULT_MAX_TOKENS,
  CRUSOE_DEFAULT_MODEL_REF,
  CRUSOE_PROVIDER_ID,
  findCrusoeStaticModel,
} from "./models.js";

function isCanonicalBaseUrl(baseUrl: string): boolean {
  return baseUrl.replace(/\/+$/, "") === CRUSOE_BASE_URL;
}

export default definePluginEntry({
  id: CRUSOE_PROVIDER_ID,
  name: "Crusoe",
  description: "Add Crusoe models to OpenClaw.",
  register(api) {
    api.registerProvider({
      id: CRUSOE_PROVIDER_ID,
      label: "Crusoe",
      envVars: ["CRUSOE_API_KEY"],
      auth: [
        createProviderApiKeyAuthMethod({
          providerId: CRUSOE_PROVIDER_ID,
          methodId: "api-key",
          label: "Crusoe API key",
          hint: "Open-weight models on Crusoe's GPU cloud",
          optionKey: "crusoeApiKey",
          flagName: "--crusoe-api-key",
          envVar: "CRUSOE_API_KEY",
          promptMessage: "Enter Crusoe API key",
          defaultModel: CRUSOE_DEFAULT_MODEL_REF,
          expectedProviders: [CRUSOE_PROVIDER_ID],
          noteTitle: "Crusoe",
          noteMessage:
            "Create a key at https://console.crusoecloud.com under Intelligence Foundry > Inference.",
        }),
      ],
      catalog: {
        order: "simple",
        run: async (ctx) => {
          const { apiKey, discoveryApiKey } = ctx.resolveProviderApiKey(CRUSOE_PROVIDER_ID);
          if (!apiKey) {
            return null;
          }
          const configuredProvider = Object.entries(ctx.config.models?.providers ?? {}).find(
            ([providerId]) => providerId.trim().toLowerCase() === CRUSOE_PROVIDER_ID,
          )?.[1];
          const baseUrl = configuredProvider?.baseUrl?.trim() || CRUSOE_BASE_URL;
          const rows = isCanonicalBaseUrl(baseUrl)
            ? await fetchCrusoeModelRows(discoveryApiKey ?? apiKey, ctx.signal)
            : undefined;
          const listed = rows ? projectCrusoeModels(rows) : [];
          return {
            provider: {
              api: "openai-completions",
              apiKey,
              baseUrl,
              models: listed.length > 0 ? listed : cloneCrusoeStaticModels(),
            },
          };
        },
      },
      staticCatalog: {
        order: "simple",
        run: async () => ({
          provider: {
            api: "openai-completions",
            baseUrl: CRUSOE_BASE_URL,
            models: cloneCrusoeStaticModels(),
          },
        }),
      },
      resolveDynamicModel: (ctx) => {
        const modelId = ctx.modelId.trim();
        if (!modelId || findCrusoeStaticModel(modelId)) {
          return undefined;
        }
        return {
          id: modelId,
          name: modelId,
          provider: CRUSOE_PROVIDER_ID,
          api: "openai-completions",
          baseUrl: CRUSOE_BASE_URL,
          reasoning: true,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: CRUSOE_DEFAULT_CONTEXT_WINDOW,
          maxTokens: CRUSOE_DEFAULT_MAX_TOKENS,
        };
      },
      isModernModelRef: () => true,
    });
  },
});
