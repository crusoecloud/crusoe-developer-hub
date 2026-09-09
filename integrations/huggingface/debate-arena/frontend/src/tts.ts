// Thin wrapper around the browser's Web Speech API. One queue, two voices (A/B),
// sentence-chunked playback so long turns don't hit Chrome's ~15s cutoff bug.
// Emits onStart once per logical utterance and onBoundary with a *global* char
// index (offset across sentence chunks) so callers can reveal text in sync.

type VoiceKind = 'a' | 'b';

interface SpeakCallbacks {
  onStart?: () => void;
  onBoundary?: (charIndex: number) => void;
  onEnd?: () => void;
}

interface QueueItem {
  text: string;
  voiceKind: VoiceKind;
  callbacks?: SpeakCallbacks;
}

class TTSManager {
  private queue: QueueItem[] = [];
  private speaking = false;
  private voiceA: SpeechSynthesisVoice | null = null;
  private voiceB: SpeechSynthesisVoice | null = null;

  constructor() {
    if (!this.supported) return;
    this.loadVoices();
    window.speechSynthesis.addEventListener('voiceschanged', () => this.loadVoices());
  }

  get supported(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  }

  private loadVoices() {
    const voices = window.speechSynthesis.getVoices();
    const en = voices.filter((v) => v.lang.toLowerCase().startsWith('en'));
    if (en.length === 0) return;
    const brighter = en.find((v) => /samantha|karen|fiona|moira|tessa|kate|susan/i.test(v.name));
    const deeper = en.find((v) => /daniel|alex|fred|oliver|tom|aaron|arthur/i.test(v.name));
    this.voiceA = brighter ?? en[0];
    this.voiceB = deeper ?? (en[1] ?? en[0]);
  }

  speak(text: string, voiceKind: VoiceKind, callbacks?: SpeakCallbacks) {
    if (!this.supported || !text.trim()) {
      callbacks?.onEnd?.();
      return;
    }
    this.queue.push({ text, voiceKind, callbacks });
    this.playNext();
  }

  private playNext() {
    if (this.speaking) return;
    const next = this.queue.shift();
    if (!next) return;
    this.speaking = true;

    const voice = next.voiceKind === 'a' ? this.voiceA : this.voiceB;

    // Chunk on sentence boundaries so each utterance stays short (<~10s),
    // sidestepping Chrome's long-utterance truncation bug. Compute each
    // sentence's offset in the original text so we can emit a global char
    // index from each utterance's onboundary event.
    const sentenceRegex = /[^.!?]+[.!?]+|\S[^.!?]*$/g;
    const matches: Array<{ text: string; start: number }> = [];
    for (const m of next.text.matchAll(sentenceRegex)) {
      if (m.index === undefined) continue;
      matches.push({ text: m[0], start: m.index });
    }
    const sentences = matches.length > 0 ? matches : [{ text: next.text, start: 0 }];

    let remaining = sentences.length;
    let startFired = false;

    const fireEnd = () => {
      this.speaking = false;
      next.callbacks?.onEnd?.();
      this.playNext();
    };

    sentences.forEach((s, i) => {
      const trimmed = s.text.trim();
      if (!trimmed) {
        remaining -= 1;
        if (remaining === 0) fireEnd();
        return;
      }
      // Align trimmed start with the original text so boundary offsets match.
      const leading = s.text.indexOf(trimmed);
      const globalStart = s.start + (leading >= 0 ? leading : 0);

      const utter = new SpeechSynthesisUtterance(trimmed);
      if (voice) utter.voice = voice;
      utter.rate = 1.0;
      utter.pitch = next.voiceKind === 'a' ? 1.08 : 0.92;

      utter.onstart = () => {
        if (!startFired) {
          startFired = true;
          next.callbacks?.onStart?.();
        }
      };

      utter.onboundary = (e: SpeechSynthesisEvent) => {
        if (!next.callbacks?.onBoundary) return;
        const localEnd = (e.charIndex ?? 0) + (e.charLength ?? 0);
        next.callbacks.onBoundary(globalStart + localEnd);
      };

      const finishOne = () => {
        // Before starting next sentence, nudge the boundary to the end of this
        // sentence so revealed text isn't stuck mid-sentence when boundary
        // events are sparse (some voices fire only on word boundaries).
        if (i === sentences.length - 1 && next.callbacks?.onBoundary) {
          next.callbacks.onBoundary(next.text.length);
        }
        remaining -= 1;
        if (remaining === 0) fireEnd();
      };
      utter.onend = finishOne;
      utter.onerror = finishOne;
      window.speechSynthesis.speak(utter);
    });
  }

  cancel() {
    this.queue = [];
    this.speaking = false;
    if (this.supported) window.speechSynthesis.cancel();
  }
}

let _instance: TTSManager | null = null;
export function getTTS(): TTSManager {
  if (!_instance) _instance = new TTSManager();
  return _instance;
}
