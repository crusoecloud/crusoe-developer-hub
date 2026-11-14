import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchCrusoeModelRows, projectCrusoeModels } from "./live-models.js";
import { CRUSOE_DEFAULT_MODEL_ID, findCrusoeStaticModel } from "./models.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

function liveRow(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    object: "model",
    is_public: true,
    type: "chat",
    context_length: 131072,
    architecture: { modality: "text", instruct_type: "chat" },
    supported_parameters: ["reasoning_effort", "tools", "max_tokens"],
    pricing: { prompt: "0.05", completion: "0.25", input_cache_reads: "0.05" },
    top_provider: { context_length: 131072, max_completion_tokens: 131072 },
    ...overrides,
  };
}

describe("Crusoe live model projection", () => {
  it("keeps only public chat deployments", () => {
    const models = projectCrusoeModels([
      liveRow({ id: "openai/gpt-oss-120b" }),
      liveRow({ id: "example/private-deployment", is_public: false, type: "" }),
      liveRow({
        id: "example/embedding-model",
        type: "embedding",
        architecture: { modality: "embedding" },
      }),
      liveRow({ id: "example/image-model", architecture: { modality: "image" } }),
      liveRow({ id: "  " }),
      "not-a-row",
      { object: "model" },
    ]);
    expect(models.map((model) => model.id)).toEqual(["openai/gpt-oss-120b"]);
  });

  it("keeps vision and untyped chat rows whatever the type field says", () => {
    const models = projectCrusoeModels([
      liveRow({
        id: "google/gemma-4-31b-it",
        type: "multimodal",
        architecture: { modality: "multimodal" },
      }),
      liveRow({ id: "nvidia/Nemotron-3.5-Lightning-30B-A3B", type: "" }),
      liveRow({ id: "example/id-only", architecture: undefined, type: undefined }),
    ]);
    expect(models.map((model) => model.id)).toEqual([
      "google/gemma-4-31b-it",
      "nvidia/Nemotron-3.5-Lightning-30B-A3B",
      "example/id-only",
    ]);
    expect(models[0]?.input).toEqual(["text", "image"]);
    expect(models[1]?.input).toEqual(["text"]);
  });

  it("overlays reviewed rows with the listed context and pricing", () => {
    const reviewed = findCrusoeStaticModel(CRUSOE_DEFAULT_MODEL_ID);
    const [model] = projectCrusoeModels([
      liveRow({
        id: CRUSOE_DEFAULT_MODEL_ID,
        context_length: 200000,
        pricing: { prompt: "0.10", completion: "0.50", input_cache_reads: "0.02" },
      }),
    ]);
    expect(model).toMatchObject({
      id: CRUSOE_DEFAULT_MODEL_ID,
      name: reviewed?.name,
      reasoning: true,
      contextWindow: 200000,
      maxTokens: reviewed?.maxTokens,
      cost: { input: 0.1, output: 0.5, cacheRead: 0.02, cacheWrite: 0 },
    });
    expect(model?.compat).toEqual(reviewed?.compat);
    expect(model?.thinkingLevelMap).toEqual(reviewed?.thinkingLevelMap);
  });

  it("projects listed-only rows from the vendor fields without an effort ladder", () => {
    const [model] = projectCrusoeModels([
      liveRow({
        id: "zai-org/GLM-5.3-Flash",
        context_length: 1048576,
        architecture: { modality: "multimodal" },
        pricing: { prompt: "0.15", completion: "0.50", input_cache_reads: "0.03" },
        top_provider: { max_completion_tokens: 131072 },
      }),
    ]);
    expect(model).toEqual({
      id: "zai-org/GLM-5.3-Flash",
      name: "zai-org/GLM-5.3-Flash",
      reasoning: true,
      input: ["text", "image"],
      cost: { input: 0.15, output: 0.5, cacheRead: 0.03, cacheWrite: 0 },
      contextWindow: 1048576,
      maxTokens: 131072,
    });
  });

  it("bounds a listed output cap by the context window and ignores bad prices", () => {
    const [model] = projectCrusoeModels([
      liveRow({
        id: "example/odd-row",
        context_length: 8192,
        supported_parameters: ["tools"],
        pricing: { prompt: "free", completion: "0.50" },
        top_provider: { max_completion_tokens: 32768 },
      }),
    ]);
    expect(model).toMatchObject({
      reasoning: false,
      maxTokens: 8192,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("keeps the first row when an id is listed twice", () => {
    const models = projectCrusoeModels([
      liveRow({ id: "example/twice", context_length: 4096 }),
      liveRow({ id: "example/twice", context_length: 8192 }),
    ]);
    expect(models).toHaveLength(1);
    expect(models[0]?.contextWindow).toBe(4096);
  });

  it("drops ids that are not plain model names", () => {
    const models = projectCrusoeModels([
      liveRow({ id: "example/plain_name-1.0:free@v2+x" }),
      liveRow({ id: "example/\u001b[31mred" }),
      liveRow({ id: "example/two words" }),
      liveRow({ id: "/leading-slash" }),
      liveRow({ id: `example/${"a".repeat(128)}` }),
    ]);
    expect(models.map((model) => model.id)).toEqual(["example/plain_name-1.0:free@v2+x"]);
  });

  it("reads no more than the first 500 rows", () => {
    const rows = Array.from({ length: 600 }, (_, index) => liveRow({ id: `example/model-${index}` }));
    const models = projectCrusoeModels(rows);
    expect(models).toHaveLength(500);
    expect(models.at(-1)?.id).toBe("example/model-499");
  });
});

describe("Crusoe live model listing", () => {
  const listing = (rows: unknown[]) => JSON.stringify({ object: "list", data: rows });

  it("returns the listed rows", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(listing([{ id: "openai/gpt-oss-120b" }]))),
    );
    expect(await fetchCrusoeModelRows("fixture-key")).toEqual([{ id: "openai/gpt-oss-120b" }]);
  });

  it("stops reading a body that grows past the size limit", async () => {
    const chunk = new Uint8Array(1024 * 1024).fill(0x20);
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        sent += 1;
        controller.enqueue(chunk);
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body)),
    );
    expect(await fetchCrusoeModelRows("fixture-key")).toBeUndefined();
    expect(sent).toBeLessThan(8);
  });

  it("does not read a body whose declared length is over the limit", async () => {
    const response = new Response(listing([]), {
      headers: { "content-length": String(4 * 1024 * 1024 + 1) },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response),
    );
    expect(await fetchCrusoeModelRows("fixture-key")).toBeUndefined();
    expect(response.bodyUsed).toBe(false);
  });
});
