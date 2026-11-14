import { readFileSync } from "node:fs";
import type {
  OpenClawPluginApi,
  ProviderCatalogContext,
  ProviderPlugin,
} from "openclaw/plugin-sdk/plugin-entry";
import { afterEach, describe, expect, it, vi } from "vitest";
import entry from "./index.js";
import {
  CRUSOE_BASE_URL,
  CRUSOE_DEFAULT_MODEL_ID,
  CRUSOE_DEFAULT_MODEL_REF,
  CRUSOE_STATIC_MODELS,
} from "./models.js";

function registerProvider(): ProviderPlugin {
  const providers: ProviderPlugin[] = [];
  const api = {
    registerProvider(provider: ProviderPlugin) {
      providers.push(provider);
    },
  } as Partial<OpenClawPluginApi>;
  entry.register(api as OpenClawPluginApi);
  const provider = providers[0];
  if (!provider || providers.length !== 1) {
    throw new Error("expected exactly one registered provider");
  }
  return provider;
}

function catalogContext(params: {
  apiKey?: string;
  discoveryApiKey?: string;
  providers?: Record<string, { baseUrl: string; models: [] }>;
}): ProviderCatalogContext {
  const credential = { apiKey: params.apiKey, discoveryApiKey: params.discoveryApiKey };
  return {
    config: { models: { providers: params.providers ?? {} } },
    env: {},
    resolveProviderApiKey: () => credential,
    resolveProviderAuth: () => ({ ...credential, mode: "api_key", source: "env" }),
  } as ProviderCatalogContext;
}

function listingResponse(rows: unknown[], init?: ResponseInit): Response {
  return new Response(JSON.stringify({ object: "list", data: rows }), init);
}

async function runCatalog(provider: ProviderPlugin, ctx: ProviderCatalogContext) {
  const result = await provider.catalog?.run(ctx);
  if (!result || !("provider" in result)) {
    return undefined;
  }
  return result.provider;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("crusoe provider plugin", () => {
  it("registers the provider with API-key auth and the default model", () => {
    const provider = registerProvider();
    expect(provider.id).toBe("crusoe");
    expect(provider.label).toBe("Crusoe");
    expect(provider.envVars).toEqual(["CRUSOE_API_KEY"]);
    expect(provider.auth).toMatchObject([
      { id: "api-key", kind: "api_key", starterModel: CRUSOE_DEFAULT_MODEL_REF },
    ]);
    expect(provider.resolveThinkingProfile).toBeUndefined();
  });

  it("returns no catalog without a key and sends no request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    expect(await registerProvider().catalog?.run(catalogContext({}))).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("lists the account's models from the Crusoe endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      listingResponse([
        {
          id: CRUSOE_DEFAULT_MODEL_ID,
          is_public: true,
          context_length: 131072,
          architecture: { modality: "text" },
          pricing: { prompt: "0.05", completion: "0.25", input_cache_reads: "0.05" },
        },
        {
          id: "zai-org/GLM-5.3",
          is_public: true,
          context_length: 1048576,
          architecture: { modality: "text" },
          supported_parameters: ["reasoning_effort", "tools"],
          pricing: { prompt: "1.40", completion: "4.40", input_cache_reads: "0.26" },
          top_provider: { max_completion_tokens: 0 },
        },
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);

    const catalog = await runCatalog(
      registerProvider(),
      catalogContext({ apiKey: "marker", discoveryApiKey: "fixture-key" }),
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${CRUSOE_BASE_URL}/models`);
    expect(init.redirect).toBe("error");
    expect(new Headers(init.headers).get("authorization")).toBe("Bearer fixture-key");
    expect(catalog).toMatchObject({
      api: "openai-completions",
      apiKey: "marker",
      baseUrl: CRUSOE_BASE_URL,
    });
    expect(catalog?.models.map((model) => model.id)).toEqual([
      CRUSOE_DEFAULT_MODEL_ID,
      "zai-org/GLM-5.3",
    ]);
    expect(catalog?.models[0]?.compat).toEqual(CRUSOE_STATIC_MODELS[0]?.compat);
    expect(catalog?.models[1]).toMatchObject({
      reasoning: true,
      contextWindow: 1048576,
      cost: { input: 1.4, output: 4.4, cacheRead: 0.26, cacheWrite: 0 },
    });
    expect(catalog?.models[1]?.compat).toBeUndefined();
  });

  it.each([
    ["a rejected key", async () => new Response("{}", { status: 401 })],
    ["a network failure", async () => Promise.reject(new Error("offline"))],
    ["an unreadable body", async () => new Response("not json")],
    ["an empty listing", async () => listingResponse([])],
  ])("keeps the reviewed rows after %s", async (_name, respond) => {
    vi.stubGlobal("fetch", vi.fn(respond));
    const catalog = await runCatalog(registerProvider(), catalogContext({ apiKey: "fixture-key" }));
    expect(catalog?.models).toEqual(CRUSOE_STATIC_MODELS);
  });

  it("does not send the key to an operator-configured base URL", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const catalog = await runCatalog(
      registerProvider(),
      catalogContext({
        apiKey: "fixture-key",
        providers: { " CRUSOE ": { baseUrl: " https://proxy.example.test/v1 ", models: [] } },
      }),
    );
    expect(fetchMock).not.toHaveBeenCalled();
    expect(catalog?.baseUrl).toBe("https://proxy.example.test/v1");
    expect(catalog?.models).toEqual(CRUSOE_STATIC_MODELS);
  });

  it.each(["http://proxy.example.test/v1", "ftp://proxy.example.test/v1", "proxy.example.test/v1"])(
    "offers no catalog for the unencrypted or malformed base URL %s",
    async (baseUrl) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const result = await registerProvider().catalog?.run(
        catalogContext({ apiKey: "fixture-key", providers: { crusoe: { baseUrl, models: [] } } }),
      );
      expect(result).toBeNull();
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it.each(["http://localhost:4000/v1", "http://127.0.0.1:4000/v1", "http://[::1]:4000/v1"])(
    "accepts the plain-http base URL %s on this machine",
    async (baseUrl) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const catalog = await runCatalog(
        registerProvider(),
        catalogContext({ apiKey: "fixture-key", providers: { crusoe: { baseUrl, models: [] } } }),
      );
      expect(fetchMock).not.toHaveBeenCalled();
      expect(catalog?.baseUrl).toBe(baseUrl);
    },
  );

  it("serves the reviewed rows as the static catalog", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const result = await registerProvider().staticCatalog?.run(catalogContext({}));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(result).toEqual({
      provider: {
        api: "openai-completions",
        baseUrl: CRUSOE_BASE_URL,
        models: CRUSOE_STATIC_MODELS,
      },
    });
  });

  it("resolves unlisted ids without copying a thinking ladder", () => {
    const provider = registerProvider();
    const resolve = (modelId: string) =>
      provider.resolveDynamicModel?.({ modelId } as Parameters<
        NonNullable<ProviderPlugin["resolveDynamicModel"]>
      >[0]);
    expect(resolve("deepseek-ai/Deepseek-V4-Flash")).toMatchObject({
      id: "deepseek-ai/Deepseek-V4-Flash",
      provider: "crusoe",
      api: "openai-completions",
      baseUrl: CRUSOE_BASE_URL,
      input: ["text"],
    });
    expect(resolve("deepseek-ai/Deepseek-V4-Flash")).not.toHaveProperty("compat");
    expect(resolve(CRUSOE_DEFAULT_MODEL_ID)).toBeUndefined();
    expect(resolve("  ")).toBeUndefined();
  });

  it("keeps the manifest catalog in step with the reviewed rows", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
    ) as {
      modelCatalog: {
        providers: { crusoe: { baseUrl: string; defaultModel: string; models: unknown[] } };
      };
    };
    const catalog = manifest.modelCatalog.providers.crusoe;
    expect(catalog.baseUrl).toBe(CRUSOE_BASE_URL);
    expect(catalog.defaultModel).toBe(CRUSOE_DEFAULT_MODEL_ID);
    expect(catalog.models).toEqual(CRUSOE_STATIC_MODELS);
  });
});
