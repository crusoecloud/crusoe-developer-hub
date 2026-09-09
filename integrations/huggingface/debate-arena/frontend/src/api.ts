import type { DebateEvent, PersonaPair } from './types';

const BASE = '/api';

export async function fetchPersonas(): Promise<{ pairs: PersonaPair[] }> {
  const res = await fetch(`${BASE}/personas`);
  if (!res.ok) throw new Error(`Failed to load personas: ${res.status}`);
  return res.json();
}

export async function* streamDebate(
  topic: string,
  rounds: number,
  personaPairId: string,
  audioEnabled: boolean,
  signal?: AbortSignal,
): AsyncGenerator<DebateEvent> {
  // When audio is driving pace, tell the server not to throttle — turns should
  // arrive fast so the TTS queue can keep up.
  const body: Record<string, unknown> = {
    topic,
    rounds,
    persona_pair_id: personaPairId,
  };
  if (audioEnabled) body.chars_per_second = 0;

  const res = await fetch(`${BASE}/debate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) throw new Error(`Debate failed: ${res.status}`);
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
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (!data) continue;
        if (data === '[DONE]') return;
        try {
          yield JSON.parse(data) as DebateEvent;
        } catch {
          /* skip malformed */
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return;
    throw err;
  }
}
