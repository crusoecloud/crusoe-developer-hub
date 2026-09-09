import { useState, useCallback, useRef } from 'react';
import type { HostConfig, TranscriptLine, Episode } from './types';
import { streamEpisode } from './api';
import InspireMe from './components/InspireMe';
import HostEditor from './components/HostEditor';
import Transcript from './components/Transcript';
import AudioPlayer from './components/AudioPlayer';
import EpisodeLibrary from './components/EpisodeLibrary';
import Subtitles from './components/Subtitles';

const DEFAULT_HOST_A: HostConfig = {
  name: 'Atlas',
  color: '#8B5CF6',
  personality:
    'Enthusiastic deep-diver. Goes on fascinating tangents, connects unexpected dots, says "actually" and "here\'s the wild part" a lot.',
};

const DEFAULT_HOST_B: HostConfig = {
  name: 'Nova',
  color: '#10B981',
  personality:
    'Sharp, witty reactor. Keeps things grounded with humor, challenges bold claims, asks the questions the audience is thinking.',
};

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function loadEpisodes(): Episode[] {
  try {
    return JSON.parse(localStorage.getItem('fakepod_episodes') || '[]');
  } catch {
    return [];
  }
}

function saveEpisodes(eps: Episode[]) {
  localStorage.setItem('fakepod_episodes', JSON.stringify(eps));
}

export default function App() {
  const [topic, setTopic] = useState('');
  const [angle, setAngle] = useState('');
  const [hostA, setHostA] = useState<HostConfig>({ ...DEFAULT_HOST_A });
  const abortRef = useRef<AbortController | null>(null);
  const [hostB, setHostB] = useState<HostConfig>({ ...DEFAULT_HOST_B });
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>(loadEpisodes);
  const [activeEpisodeId, setActiveEpisodeId] = useState<string | null>(null);
  const [showHosts, setShowHosts] = useState(false);
  const [showInspire, setShowInspire] = useState(false);
  const [currentTopic, setCurrentTopic] = useState('');
  const [audioPlaying, setAudioPlaying] = useState(false);

  const handleGenerate = useCallback(async () => {
    const t = topic.trim();
    if (!t || isGenerating) return;

    const controller = new AbortController();
    abortRef.current = controller;

    setIsGenerating(true);
    setTranscript([]);
    setCurrentTopic(t);
    setActiveEpisodeId(null);

    try {
      const stream = streamEpisode(
        t,
        angle.trim() || undefined,
        0, // continuous mode — runs until stopped
        { name: hostA.name, personality: hostA.personality },
        { name: hostB.name, personality: hostB.personality },
        controller.signal,
      );

      const lines: TranscriptLine[] = [];
      for await (const line of stream) {
        lines.push(line);
        setTranscript([...lines]);
      }

      // Auto-save episode
      const ep: Episode = {
        id: generateId(),
        topic: t,
        angle: angle.trim() || undefined,
        transcript: lines,
        createdAt: new Date().toISOString(),
        hostA,
        hostB,
      };
      const updated = [ep, ...episodes];
      setEpisodes(updated);
      saveEpisodes(updated);
      setActiveEpisodeId(ep.id);
    } catch (err) {
      console.error('Generation failed:', err);
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [topic, angle, hostA, hostB, isGenerating, episodes]);

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);


  const handleSelectEpisode = (ep: Episode) => {
    setActiveEpisodeId(ep.id);
    setTranscript(ep.transcript);
    setTopic(ep.topic);
    setAngle(ep.angle || '');
    setCurrentTopic(ep.topic);
    setHostA(ep.hostA);
    setHostB(ep.hostB);
  };

  const handleDeleteEpisode = (id: string) => {
    const updated = episodes.filter((ep) => ep.id !== id);
    setEpisodes(updated);
    saveEpisodes(updated);
    if (activeEpisodeId === id) {
      setActiveEpisodeId(null);
      setTranscript([]);
    }
  };

  const handleInspireSelect = (idea: string) => {
    setTopic(idea);
    setShowInspire(false);
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-200 px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-2xl">&#127897;</span>
          <h1 className="text-xl font-bold tracking-tight">
            Fake<span className="text-purple-600">Pod</span>
          </h1>
          <span className="text-xs text-gray-500 hidden sm:inline">
            AI Podcast Generator &middot; Powered by Crusoe Cloud Foundry
          </span>
        </div>
        <button
          onClick={() => setShowHosts(!showHosts)}
          className="btn-secondary text-sm"
        >
          {showHosts ? 'Hide Hosts' : '\u2699 Hosts'}
        </button>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 border-r border-gray-200 flex flex-col shrink-0 hidden lg:flex">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
              Episodes
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto p-3">
            <EpisodeLibrary
              episodes={episodes}
              activeId={activeEpisodeId}
              onSelect={handleSelectEpisode}
              onDelete={handleDeleteEpisode}
            />
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Host editors (collapsible) */}
          {showHosts && (
            <div className="border-b border-gray-200 p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              <HostEditor label="Host A — Qwen3-235B" host={hostA} onChange={setHostA} />
              <HostEditor label="Host B — Nemotron-120B" host={hostB} onChange={setHostB} />
            </div>
          )}

          {/* Transcript area */}
          <div className="flex-1 flex flex-col overflow-hidden p-4 gap-3">
            {/* Audio player */}
            {transcript.length > 0 && (
              <AudioPlayer transcript={transcript} onLineActive={setActiveLine} onPlayingChange={setAudioPlaying} />
            )}

            {/* Subtitles */}
            {transcript.length > 0 && (
              <Subtitles
                lines={transcript}
                audioActiveLine={activeLine}
                audioPlaying={audioPlaying}
                isGenerating={isGenerating}
              />
            )}

            {/* Episode title */}
            {currentTopic && transcript.length > 0 && (
              <h2 className="text-lg font-semibold text-gray-800 px-1">
                &ldquo;{currentTopic}&rdquo;
              </h2>
            )}

            {/* Transcript */}
            <Transcript
              lines={transcript}
              isGenerating={isGenerating}
              activeLine={activeLine}
            />

          </div>

          {/* Input bar */}
          <div className="border-t border-gray-200 p-4 space-y-3 shrink-0">
            {/* Inspire me toggle */}
            {showInspire && (
              <div className="card">
                <InspireMe onSelect={handleInspireSelect} />
              </div>
            )}

            <div className="flex gap-2">
              <div className="flex-1 space-y-2">
                <div className="flex gap-2">
                  <input
                    className="input-field"
                    placeholder="What's your episode about?"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !isGenerating && handleGenerate()}
                    disabled={isGenerating}
                  />
                  <input
                    className="input-field w-48 hidden md:block"
                    placeholder="Angle (optional)"
                    value={angle}
                    onChange={(e) => setAngle(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !isGenerating && handleGenerate()}
                    disabled={isGenerating}
                  />
                </div>
                {isGenerating && (
                  <p className="text-xs text-gray-400">
                    Conversation is live — hosts will keep talking until you stop them.
                  </p>
                )}
              </div>

              <div className="flex flex-col gap-2 shrink-0">
                {isGenerating ? (
                  <button
                    onClick={handleStop}
                    className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors h-full"
                  >
                    {'\u23F9'} Stop
                  </button>
                ) : (
                  <button
                    onClick={handleGenerate}
                    disabled={!topic.trim()}
                    className="btn-primary h-full"
                  >
                    Generate {'\u25B6'}
                  </button>
                )}
                <button
                  onClick={() => setShowInspire(!showInspire)}
                  className="btn-secondary text-sm"
                  disabled={isGenerating}
                >
                  &#127922; Inspire Me
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
