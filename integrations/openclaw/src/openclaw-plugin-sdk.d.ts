declare module "openclaw/plugin-sdk/plugin-entry" {
  export type ModelApi =
    | "openai-completions"
    | "openai-responses"
    | "openai-chatgpt-responses"
    | "anthropic-messages"
    | "google-generative-ai"
    | "google-interactions"
    | "google-vertex"
    | "github-copilot"
    | "bedrock-converse-stream"
    | "ollama"
    | "pi-messages"
    | "azure-openai-responses";

  export type ModelThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

  export type ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>;

  export type ModelCostConfig = {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };

  export type ModelCompatConfig = {
    supportsUsageInStreaming?: boolean;
    supportsPromptCacheKey?: boolean;
    supportsStrictMode?: boolean;
    supportsReasoningEffort?: boolean;
    supportedReasoningEfforts?: string[];
    reasoningEffortMap?: Record<string, string>;
    thinkingLevelLabels?: Record<string, string>;
    defaultThinkingLevel?: string;
  };

  export type ModelDefinitionConfig = {
    id: string;
    name: string;
    api?: ModelApi;
    baseUrl?: string;
    reasoning: boolean;
    thinkingLevelMap?: ThinkingLevelMap;
    input: ("text" | "image")[];
    cost: ModelCostConfig;
    contextWindow?: number;
    maxTokens: number;
    compat?: ModelCompatConfig;
  };

  export type ModelProviderConfig = {
    baseUrl: string;
    apiKey?: string;
    api?: ModelApi;
    models: ModelDefinitionConfig[];
  };

  export type OpenClawConfig = {
    models?: {
      providers?: Record<string, ModelProviderConfig>;
    };
  };

  export type ProviderAuthKind = "oauth" | "api_key" | "token" | "device_code" | "custom";

  export type ProviderAuthMethod = {
    id: string;
    label: string;
    hint?: string;
    kind: ProviderAuthKind;
    starterModel?: string;
    run: (ctx: unknown) => Promise<unknown>;
    runNonInteractive?: (ctx: unknown) => Promise<OpenClawConfig | null>;
    validateNonInteractive?: (ctx: unknown) => Promise<boolean>;
  };

  export type ProviderCatalogOrder = "simple" | "profile" | "paired" | "late";

  export type ProviderCatalogContext = {
    signal?: AbortSignal;
    config: OpenClawConfig;
    agentDir?: string;
    workspaceDir?: string;
    env: NodeJS.ProcessEnv;
    providerIds?: readonly string[];
    resolveProviderApiKey: (providerId?: string) => {
      apiKey: string | undefined;
      discoveryApiKey?: string;
      profileId?: string;
      mode?: "api_key" | "oauth" | "token";
    };
    resolveProviderAuth: (
      providerId?: string,
      options?: { oauthMarker?: string },
    ) => {
      apiKey: string | undefined;
      discoveryApiKey?: string;
      mode: "api_key" | "aws-sdk" | "oauth" | "token" | "none";
      source: "env" | "profile" | "none";
      profileId?: string;
    };
  };

  export type ProviderCatalogResult =
    | { provider: ModelProviderConfig }
    | { providers: Record<string, ModelProviderConfig> }
    | null
    | undefined;

  export type ProviderPluginCatalog = {
    order?: ProviderCatalogOrder;
    run: (ctx: ProviderCatalogContext) => Promise<ProviderCatalogResult>;
  };

  export type ProviderResolveDynamicModelContext = {
    config?: OpenClawConfig;
    agentDir?: string;
    workspaceDir?: string;
    provider: string;
    modelId: string;
  };

  export type ProviderRuntimeModel = {
    id: string;
    name: string;
    api: ModelApi;
    provider: string;
    baseUrl: string;
    reasoning: boolean;
    thinkingLevelMap?: ThinkingLevelMap;
    input: ("text" | "image")[];
    cost: ModelCostConfig;
    contextWindow?: number;
    maxTokens: number;
    compat?: ModelCompatConfig;
  };

  export type ProviderModernModelPolicyContext = {
    provider: string;
    modelId: string;
  };

  export type ProviderPlugin = {
    id: string;
    label: string;
    envVars?: string[];
    auth: ProviderAuthMethod[];
    catalog?: ProviderPluginCatalog;
    staticCatalog?: ProviderPluginCatalog;
    resolveDynamicModel?: (
      ctx: ProviderResolveDynamicModelContext,
    ) => ProviderRuntimeModel | null | undefined;
    resolveThinkingProfile?: (ctx: unknown) => unknown;
    isModernModelRef?: (ctx: ProviderModernModelPolicyContext) => boolean | undefined;
  };

  export type OpenClawPluginApi = {
    registerProvider: (provider: ProviderPlugin) => void;
  };

  export type DefinePluginEntryOptions = {
    id: string;
    name: string;
    description: string;
    register: (api: OpenClawPluginApi) => void | Promise<void>;
  };

  export type DefinedPluginEntry = DefinePluginEntryOptions;

  export function definePluginEntry(options: DefinePluginEntryOptions): DefinedPluginEntry;
}

declare module "openclaw/plugin-sdk/provider-auth" {
  import type { OpenClawConfig, ProviderAuthMethod } from "openclaw/plugin-sdk/plugin-entry";

  export type ProviderApiKeyAuthMethodOptions = {
    providerId: string;
    methodId: string;
    label: string;
    hint?: string;
    optionKey: string;
    flagName: `--${string}`;
    envVar: string;
    promptMessage: string;
    validateApiKey?: (apiKey: string) => string | undefined;
    profileId?: string;
    profileIds?: string[];
    allowProfile?: boolean;
    defaultModel?: string;
    preserveExistingPrimary?: boolean;
    expectedProviders?: string[];
    metadata?: Record<string, string>;
    noteMessage?: string;
    noteTitle?: string;
    applyConfig?: (cfg: OpenClawConfig) => OpenClawConfig;
  };

  export function createProviderApiKeyAuthMethod(
    params: ProviderApiKeyAuthMethodOptions,
  ): ProviderAuthMethod;
}
