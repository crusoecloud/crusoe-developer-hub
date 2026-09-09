import type { PersonaPair } from '../types';

interface Props {
  topic: string;
  onTopicChange: (v: string) => void;
  rounds: number;
  onRoundsChange: (n: number) => void;
  pairs: PersonaPair[];
  selectedPairId: string;
  onSelectPair: (id: string) => void;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
  audioEnabled: boolean;
  onAudioToggle: (enabled: boolean) => void;
  audioSupported: boolean;
}

const SUGGESTIONS = [
  'Should AI have the right to create art?',
  'Is working from home better than the office?',
  'Is pineapple acceptable on pizza?',
  'Will humans colonize Mars within 50 years?',
  'Are video games a legitimate art form?',
];

export default function Controls({
  topic,
  onTopicChange,
  rounds,
  onRoundsChange,
  pairs,
  selectedPairId,
  onSelectPair,
  running,
  onStart,
  onStop,
  audioEnabled,
  onAudioToggle,
  audioSupported,
}: Props) {
  const canStart = topic.trim().length > 0 && !running;

  return (
    <div className="w-full max-w-5xl mx-auto bg-slate-900/60 backdrop-blur border border-white/10 rounded-2xl p-5 flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-slate-300">Debate topic</label>
        <input
          type="text"
          value={topic}
          onChange={(e) => onTopicChange(e.target.value)}
          placeholder="e.g. Should self-driving cars be held to a higher safety bar than humans?"
          disabled={running}
          className="w-full bg-slate-800/70 text-slate-100 rounded-lg px-4 py-2.5 border border-white/10 focus:border-indigo-400 focus:outline-none disabled:opacity-50"
        />
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onTopicChange(s)}
              disabled={running}
              className="text-xs px-2.5 py-1 rounded-full bg-white/5 hover:bg-white/10 text-slate-300 disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 flex flex-col gap-2">
          <label className="text-sm font-medium text-slate-300">Personas</label>
          <select
            value={selectedPairId}
            onChange={(e) => onSelectPair(e.target.value)}
            disabled={running || pairs.length === 0}
            className="w-full bg-slate-800/70 text-slate-100 rounded-lg px-3 py-2.5 border border-white/10 focus:border-indigo-400 focus:outline-none disabled:opacity-50"
          >
            {pairs.map((p) => (
              <option key={p.id} value={p.id}>
                {p.a.emoji} {p.a.name} vs {p.b.name} {p.b.emoji} — {p.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1 flex flex-col gap-2">
          <label className="text-sm font-medium text-slate-300">
            Rounds: <span className="text-indigo-300 font-bold">{rounds}</span>
            <span className="text-slate-500 text-xs ml-2">({rounds * 2} total turns)</span>
          </label>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={rounds}
            onChange={(e) => onRoundsChange(parseInt(e.target.value, 10))}
            disabled={running}
            className="w-full accent-indigo-400"
          />
        </div>

        <div className="flex items-end gap-2">
          <button
            type="button"
            onClick={() => onAudioToggle(!audioEnabled)}
            disabled={!audioSupported}
            title={
              !audioSupported
                ? 'Web Speech API not available in this browser'
                : audioEnabled
                  ? 'Mute audio'
                  : 'Enable audio'
            }
            aria-label={audioEnabled ? 'Mute audio' : 'Enable audio'}
            className="px-3 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {audioEnabled && audioSupported ? '🔊' : '🔇'}
          </button>
          {!running ? (
            <button
              type="button"
              onClick={onStart}
              disabled={!canStart}
              className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-indigo-500 to-pink-500 text-white font-semibold shadow-lg hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Start Debate
            </button>
          ) : (
            <button
              type="button"
              onClick={onStop}
              className="px-6 py-2.5 rounded-lg bg-red-500/90 text-white font-semibold shadow-lg hover:brightness-110"
            >
              Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
