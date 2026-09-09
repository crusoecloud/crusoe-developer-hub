import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchPersonas, streamDebate } from './api';
import Controls from './components/Controls';
import Stage from './components/Stage';
import TranscriptLog from './components/TranscriptLog';
import { getTTS } from './tts';
import type { PersonaMini, PersonaPair, Speaker, Turn } from './types';

const AUDIO_PREF_KEY = 'debate-arena.audio-enabled';

export default function App() {
  const [pairs, setPairs] = useState<PersonaPair[]>([]);
  const [selectedPairId, setSelectedPairId] = useState<string>('optimist-skeptic');
  const [topic, setTopic] = useState('');
  const [rounds, setRounds] = useState(3);

  const [personaA, setPersonaA] = useState<PersonaMini | null>(null);
  const [personaB, setPersonaB] = useState<PersonaMini | null>(null);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [currentTextA, setCurrentTextA] = useState('');
  const [currentTextB, setCurrentTextB] = useState('');
  const [activeSpeaker, setActiveSpeaker] = useState<Speaker | null>(null);
  const [turnKeyA, setTurnKeyA] = useState(0);
  const [turnKeyB, setTurnKeyB] = useState(0);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioEnabled, setAudioEnabled] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true;
    const stored = window.localStorage.getItem(AUDIO_PREF_KEY);
    return stored === null ? true : stored === 'true';
  });

  const abortRef = useRef<AbortController | null>(null);
  const prevSpeakerRef = useRef<Speaker | null>(null);
  const tts = getTTS();

  useEffect(() => {
    window.localStorage.setItem(AUDIO_PREF_KEY, String(audioEnabled));
    if (!audioEnabled) tts.cancel();
  }, [audioEnabled, tts]);

  // Load persona pairs on mount
  useEffect(() => {
    fetchPersonas()
      .then((r) => {
        setPairs(r.pairs);
        if (r.pairs.length > 0) {
          const first = r.pairs[0];
          setSelectedPairId((cur) => (r.pairs.some((p) => p.id === cur) ? cur : first.id));
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Preview personas when selection changes (before debate starts)
  useEffect(() => {
    const p = pairs.find((x) => x.id === selectedPairId);
    if (p && !running) {
      setPersonaA(p.a);
      setPersonaB(p.b);
    }
  }, [selectedPairId, pairs, running]);

  const handleStart = useCallback(async () => {
    if (!topic.trim()) return;

    setError(null);
    setTurns([]);
    setCurrentTextA('');
    setCurrentTextB('');
    setActiveSpeaker(null);
    setTurnKeyA(0);
    setTurnKeyB(0);
    prevSpeakerRef.current = null;
    tts.cancel();
    setRunning(true);

    // Capture audio mode for the duration of this debate; toggling mid-stream
    // would make the text/audio flow inconsistent.
    const audioMode = audioEnabled && tts.supported;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const evt of streamDebate(
        topic,
        rounds,
        selectedPairId,
        audioMode,
        controller.signal,
      )) {
        if ('error' in evt) {
          setError(evt.error);
          break;
        } else if ('event' in evt) {
          setPersonaA(evt.persona_a);
          setPersonaB(evt.persona_b);
        } else if (evt.done) {
          const speaker = evt.speaker;
          const finalText = evt.text;
          setTurns((prev) => [
            ...prev,
            {
              speaker,
              name: evt.name,
              color: evt.color,
              text: finalText,
              round: evt.round,
              done: true,
            },
          ]);
          if (audioMode) {
            // Reveal is driven by TTS boundaries — don't touch the bubble here.
            tts.speak(finalText, speaker, {
              onStart: () => {
                setActiveSpeaker(speaker);
                if (speaker === 'a') {
                  setCurrentTextA('');
                  setTurnKeyA((k) => k + 1);
                } else {
                  setCurrentTextB('');
                  setTurnKeyB((k) => k + 1);
                }
              },
              onBoundary: (idx) => {
                const revealed = finalText.slice(0, idx);
                if (speaker === 'a') setCurrentTextA(revealed);
                else setCurrentTextB(revealed);
              },
              onEnd: () => {
                if (speaker === 'a') setCurrentTextA(finalText);
                else setCurrentTextB(finalText);
                setActiveSpeaker(null);
              },
            });
          } else {
            if (speaker === 'a') setCurrentTextA(finalText);
            else setCurrentTextB(finalText);
            setActiveSpeaker(speaker);
          }
        } else {
          const speaker = evt.speaker;
          if (audioMode) {
            // Hold the text silently. Bubble reveal happens in TTS callbacks.
            continue;
          }
          // Audio off: stream chunks directly into the bubble at server pace.
          const isNewTurn = prevSpeakerRef.current !== speaker;
          prevSpeakerRef.current = speaker;

          if (speaker === 'a') {
            if (isNewTurn) {
              setCurrentTextA(evt.chunk);
              setTurnKeyA((k) => k + 1);
            } else {
              setCurrentTextA((prev) => prev + evt.chunk);
            }
          } else {
            if (isNewTurn) {
              setCurrentTextB(evt.chunk);
              setTurnKeyB((k) => k + 1);
            } else {
              setCurrentTextB((prev) => prev + evt.chunk);
            }
          }
          setActiveSpeaker(speaker);
        }
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setError(String(err));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
      // If audio is off, clear active speaker now. If audio is on, the TTS
      // queue may still be playing; let its onEnd callbacks clear it.
      if (!audioMode) setActiveSpeaker(null);
    }
  }, [topic, rounds, selectedPairId, audioEnabled, tts]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    tts.cancel();
  }, [tts]);

  return (
    <div className="min-h-screen flex flex-col text-slate-100">
      <header className="w-full py-6 px-8 text-center">
        <h1 className="text-3xl md:text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-pink-300">
          Debate Arena
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Two Gemma 4 personas argue it out — powered by Crusoe Foundry
        </p>
      </header>

      <main className="flex-1 flex flex-col gap-6 pb-10 px-4">
        <Controls
          topic={topic}
          onTopicChange={setTopic}
          rounds={rounds}
          onRoundsChange={setRounds}
          pairs={pairs}
          selectedPairId={selectedPairId}
          onSelectPair={setSelectedPairId}
          running={running}
          onStart={handleStart}
          onStop={handleStop}
          audioEnabled={audioEnabled}
          onAudioToggle={setAudioEnabled}
          audioSupported={tts.supported}
        />

        {error && (
          <div className="max-w-5xl mx-auto w-full bg-red-500/10 border border-red-400/30 text-red-200 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <Stage
          personaA={personaA}
          personaB={personaB}
          activeSpeaker={activeSpeaker}
          isStreaming={activeSpeaker !== null}
          currentTextA={currentTextA}
          currentTextB={currentTextB}
          bubbleVisibleA={currentTextA.length > 0 || activeSpeaker === 'a'}
          bubbleVisibleB={currentTextB.length > 0 || activeSpeaker === 'b'}
          turnKeyA={turnKeyA}
          turnKeyB={turnKeyB}
        />

        <TranscriptLog turns={turns} />
      </main>

      <footer className="py-4 text-center text-xs text-slate-500">
        Built with Crusoe Foundry · google/gemma-4-31b-it · Deployed on HuggingFace Spaces
      </footer>
    </div>
  );
}
