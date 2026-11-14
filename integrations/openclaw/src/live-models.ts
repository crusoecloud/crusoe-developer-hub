import {
  CRUSOE_BASE_URL,
  CRUSOE_DEFAULT_CONTEXT_WINDOW,
  CRUSOE_DEFAULT_MAX_TOKENS,
  findCrusoeStaticModel,
  type CrusoeModel,
} from "./models.js";

const LIST_TIMEOUT_MS = 5_000;
const LIST_MAX_BYTES = 4 * 1024 * 1024;
const LIST_MAX_ROWS = 500;
const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function readLowercaseString(value: unknown): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function readPositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function readPricePerMillion(value: unknown): number | undefined {
  const text = typeof value === "number" ? String(value) : typeof value === "string" ? value : "";
  if (!/^\s*\d+(?:\.\d+)?\s*$/.test(text)) {
    return undefined;
  }
  const price = Number(text);
  return Number.isFinite(price) && price >= 0 ? price : undefined;
}

function isPublicChatRow(row: Record<string, unknown>): boolean {
  if (row.is_public === false) {
    return false;
  }
  const architecture = isRecord(row.architecture) ? row.architecture : undefined;
  const modality = readLowercaseString(architecture?.modality);
  return modality === "" || modality === "text" || modality === "multimodal";
}

function projectRow(row: Record<string, unknown>, id: string): CrusoeModel {
  const reviewed = findCrusoeStaticModel(id);
  const architecture = isRecord(row.architecture) ? row.architecture : undefined;
  const topProvider = isRecord(row.top_provider) ? row.top_provider : undefined;
  const pricing = isRecord(row.pricing) ? row.pricing : undefined;

  const contextWindow =
    readPositiveInteger(row.context_length) ??
    reviewed?.contextWindow ??
    CRUSOE_DEFAULT_CONTEXT_WINDOW;
  const maxTokens =
    reviewed?.maxTokens ??
    Math.min(
      readPositiveInteger(topProvider?.max_completion_tokens) ?? CRUSOE_DEFAULT_MAX_TOKENS,
      contextWindow,
    );
  const input = readPricePerMillion(pricing?.prompt);
  const output = readPricePerMillion(pricing?.completion);
  const cacheRead = readPricePerMillion(pricing?.input_cache_reads);
  const cost =
    input !== undefined && output !== undefined
      ? { input, output, cacheRead: cacheRead ?? input, cacheWrite: 0 }
      : (reviewed?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  const acceptsImages =
    readLowercaseString(architecture?.modality) === "multimodal" ||
    reviewed?.input.includes("image") === true;
  const inputKinds: CrusoeModel["input"] = acceptsImages ? ["text", "image"] : ["text"];

  if (reviewed) {
    return { ...structuredClone(reviewed), contextWindow, maxTokens, cost, input: inputKinds };
  }
  const parameters = Array.isArray(row.supported_parameters) ? row.supported_parameters : [];
  return {
    id,
    name: id,
    reasoning: parameters.includes("reasoning_effort"),
    input: inputKinds,
    cost,
    contextWindow,
    maxTokens,
  };
}

export function projectCrusoeModels(rows: readonly unknown[]): CrusoeModel[] {
  const models = new Map<string, CrusoeModel>();
  for (const row of rows.slice(0, LIST_MAX_ROWS)) {
    if (!isRecord(row) || !isPublicChatRow(row)) {
      continue;
    }
    const id = typeof row.id === "string" ? row.id.trim() : "";
    if (MODEL_ID_PATTERN.test(id) && !models.has(id)) {
      models.set(id, projectRow(row, id));
    }
  }
  return [...models.values()];
}

async function readBoundedText(response: Response): Promise<string | undefined> {
  if (Number(response.headers.get("content-length") ?? "") > LIST_MAX_BYTES) {
    return undefined;
  }
  const reader = response.body?.getReader();
  if (!reader) {
    return undefined;
  }
  const decoder = new TextDecoder();
  let received = 0;
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      return text + decoder.decode();
    }
    received += value.byteLength;
    if (received > LIST_MAX_BYTES) {
      await reader.cancel();
      return undefined;
    }
    text += decoder.decode(value, { stream: true });
  }
}

export async function fetchCrusoeModelRows(
  apiKey: string,
  signal?: AbortSignal,
): Promise<unknown[] | undefined> {
  const timeout = AbortSignal.timeout(LIST_TIMEOUT_MS);
  try {
    const response = await fetch(`${CRUSOE_BASE_URL}/models`, {
      headers: { Accept: "application/json", Authorization: `Bearer ${apiKey}` },
      redirect: "error",
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    if (!response.ok) {
      return undefined;
    }
    const text = await readBoundedText(response);
    if (text === undefined) {
      return undefined;
    }
    const body: unknown = JSON.parse(text);
    return isRecord(body) && Array.isArray(body.data) ? body.data : undefined;
  } catch {
    return undefined;
  }
}
