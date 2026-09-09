import type { TranscriptLine } from './types';

const BASE = '/api';

export async function fetchDefaultHosts() {
  const res = await fetch(`${BASE}/hosts`);
  return res.json();
}

export async function fetchCategories(): Promise<{ categories: string[] }> {
  const res = await fetch(`${BASE}/categories`);
  return res.json();
}

export async function fetchInspiration(category?: string): Promise<{ ideas: string[] }> {
  const res = await fetch(`${BASE}/inspire`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category: category || 'all' }),
  });
  return res.json();
}

export async function* streamEpisode(
  topic: string,
  angle: string | undefined,
  rounds: number,
  hostA?: { name: string; personality: string },
  hostB?: { name: string; personality: string },
  signal?: AbortSignal,
): AsyncGenerator<TranscriptLine> {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      topic,
      angle: angle || null,
      rounds,
      host_a: hostA || null,
      host_b: hostB || null,
    }),
    signal,
  });

  if (!res.ok) throw new Error(`Generate failed: ${res.status}`);
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              console.error('Stream error:', parsed.error);
              return;
            }
            yield parsed as TranscriptLine;
          } catch {
            // skip malformed
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      // User stopped the generation — clean exit
      return;
    }
    throw err;
  }
}

export async function fetchTTS(text: string, host: 'a' | 'b'): Promise<Blob> {
  const res = await fetch(`${BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, host }),
  });
  if (!res.ok) throw new Error(`TTS failed: ${res.status}`);
  return res.blob();
}
