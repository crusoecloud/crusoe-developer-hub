import { useState, useEffect, useRef, useCallback } from 'react';
import type { TranscriptLine } from '../types';
import { fetchTTS } from '../api';

type TTSMode = 'server' | 'browser' | 'testing';

interface Props {
  transcript: TranscriptLine[];
  onLineActive: (index: number | null) => void;
  onPlayingChange: (playing: boolean) => void;
}

/**
 * Pick two distinct browser voices — tries to find one male-ish and one female-ish,
 * falls back to first two available.
 */
function pickBrowserVoices(): { voiceA: SpeechSynthesisVoice | null; voiceB: SpeechSynthesisVoice | null } {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'));
  if (voices.length === 0) return { voiceA: null, voiceB: null };
  if (voices.length === 1) return { voiceA: voices[0], voiceB: voices[0] };

  // Try to find distinct voices
  const male = voices.find((v) => /male|guy|chris|david|james|mark|ryan/i.test(v.name));
  const female = voices.find((v) => /female|woman|samantha|aria|jenny|zira|sonia/i.test(v.name));

  return {
    voiceA: male || voices[0],
    voiceB: female || voices[1] || voices[0],
  };
}

export default function AudioPlayer({ transcript, onLineActive, onPlayingChange }: Props) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentLine, setCurrentLine] = useState(0);
  const [loading, setLoading] = useState(false);
  const [ttsMode, setTtsMode] = useState<TTSMode>('testing');

  const audioCache = useRef<Map<number, Blob>>(new Map());
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlsRef = useRef<string[]>([]);
  const playingRef = useRef(false);
  const prefetchedRef = useRef<Set<number>>(new Set());
  const browserVoices = useRef<{ voiceA: SpeechSynthesisVoice | null; voiceB: SpeechSynthesisVoice | null }>({ voiceA: null, voiceB: null });

  // Test if server TTS works on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/tts/test');
        const data = await res.json();
        if (!cancelled) {
          setTtsMode(data.available ? 'server' : 'browser');
        }
      } catch {
        if (!cancelled) setTtsMode('browser');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Load browser voices (they sometimes load async)
  useEffect(() => {
    const loadVoices = () => {
      browserVoices.current = pickBrowserVoices();
    };
    loadVoices();
    speechSynthesis.addEventListener('voiceschanged', loadVoices);
    return () => speechSynthesis.removeEventListener('voiceschanged', loadVoices);
  }, []);

  useEffect(() => {
    onPlayingChange(isPlaying);
  }, [isPlaying, onPlayingChange]);

  useEffect(() => {
    return () => {
      urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      speechSynthesis.cancel();
    };
  }, []);

  // Pre-fetch server TTS for new lines
  useEffect(() => {
    if (ttsMode !== 'server') return;
    transcript.forEach((line, i) => {
      if (!prefetchedRef.current.has(i)) {
        prefetchedRef.current.add(i);
        fetchTTS(line.text, line.host)
          .then((blob) => audioCache.current.set(i, blob))
          .catch(() => { /* will use browser fallback for this line */ });
      }
    });
  }, [transcript, ttsMode]);

  // Reset when transcript clears
  useEffect(() => {
    if (transcript.length === 0) {
      prefetchedRef.current.clear();
      audioCache.current.clear();
    }
  }, [transcript.length]);

  /** Play a single line via server audio */
  const playLineServer = useCallback(async (index: number): Promise<boolean> => {
    // Wait for blob to arrive in cache
    let waited = 0;
    while (!audioCache.current.has(index) && playingRef.current && waited < 15000) {
      setLoading(true);
      await new Promise((r) => setTimeout(r, 300));
      waited += 300;
    }
    setLoading(false);
    if (!playingRef.current) return false;

    const blob = audioCache.current.get(index);
    if (!blob) return true; // skip failed lines

    const url = URL.createObjectURL(blob);
    urlsRef.current.push(url);
    const audio = new Audio(url);
    audioRef.current = audio;
    setCurrentLine(index);
    onLineActive(index);

    return new Promise<boolean>((resolve) => {
      audio.onended = () => resolve(true);
      audio.onerror = () => resolve(true);
      audio.play().catch(() => resolve(true));
    });
  }, [onLineActive]);

  /** Play a single line via browser SpeechSynthesis */
  const playLineBrowser = useCallback((index: number): Promise<boolean> => {
    if (index >= transcript.length) return Promise.resolve(false);
    const line = transcript[index];
    setCurrentLine(index);
    onLineActive(index);

    return new Promise<boolean>((resolve) => {
      const utterance = new SpeechSynthesisUtterance(line.text);
      utterance.rate = 1.05;
      utterance.pitch = line.host === 'a' ? 0.9 : 1.15;
      const { voiceA, voiceB } = browserVoices.current;
      utterance.voice = line.host === 'a' ? voiceA : voiceB;
      utterance.onend = () => resolve(true);
      utterance.onerror = () => resolve(true);
      speechSynthesis.speak(utterance);
    });
  }, [transcript, onLineActive]);

  const startPlaying = useCallback(async (fromIndex: number) => {
    playingRef.current = true;
    setIsPlaying(true);

    let i = fromIndex;
    while (playingRef.current) {
      // Wait for the line to exist in transcript
      if (i >= transcript.length) {
        setLoading(true);
        let waited = 0;
        while (i >= transcript.length && playingRef.current && waited < 30000) {
          await new Promise((r) => setTimeout(r, 500));
          waited += 500;
          // Re-check via closure — transcript.length won't update, but for
          // server mode the cache will have the entry
          if (ttsMode === 'server' && audioCache.current.has(i)) break;
        }
        setLoading(false);
        // For browser mode we can't proceed without the transcript line
        if (ttsMode === 'browser' && i >= transcript.length) break;
        if (ttsMode === 'server' && !audioCache.current.has(i)) break;
        if (!playingRef.current) break;
      }

      const ok = ttsMode === 'server'
        ? await playLineServer(i)
        : await playLineBrowser(i);

      if (!ok || !playingRef.current) break;
      i++;
    }

    playingRef.current = false;
    setIsPlaying(false);
    onLineActive(null);
  }, [ttsMode, transcript, playLineServer, playLineBrowser, onLineActive]);

  const togglePlay = () => {
    if (transcript.length === 0 || ttsMode === 'testing') return;
    if (isPlaying) {
      stopPlayback();
      return;
    }
    startPlaying(currentLine);
  };

  const stopPlayback = () => {
    playingRef.current = false;
    speechSynthesis.cancel();
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsPlaying(false);
    setCurrentLine(0);
    onLineActive(null);
  };

  const progress = transcript.length > 0 ? ((currentLine + 1) / transcript.length) * 100 : 0;
  const modeLabel = ttsMode === 'server'
    ? '\u{1F3A4} AI Voices'
    : ttsMode === 'browser'
      ? '\u{1F3A4} Browser Voices'
      : '\u{1F3A4} Checking TTS...';

  return (
    <div className="card flex items-center gap-3">
      <button
        onClick={togglePlay}
        disabled={transcript.length === 0 || ttsMode === 'testing'}
        className="w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-500 disabled:opacity-30 flex items-center justify-center transition-colors shrink-0"
      >
        {isPlaying ? (
          <span className="text-white text-sm font-bold">&#9646;&#9646;</span>
        ) : (
          <span className="text-white ml-0.5">&#9654;</span>
        )}
      </button>

      <div className="flex-1 space-y-1">
        <div className="w-full h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-purple-500 rounded-full transition-all duration-300"
            style={{ width: `${isPlaying ? progress : 0}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500">
          <span>
            {loading
              ? 'Loading audio...'
              : isPlaying
                ? `Line ${currentLine + 1} / ${transcript.length}`
                : transcript.length > 0
                  ? 'Press play to listen'
                  : 'Ready'}
          </span>
          <span className="text-purple-500">{modeLabel}</span>
        </div>
      </div>

      {isPlaying && (
        <button onClick={stopPlayback} className="text-gray-400 hover:text-gray-900 transition-colors">
          &#9632;
        </button>
      )}
    </div>
  );
}
