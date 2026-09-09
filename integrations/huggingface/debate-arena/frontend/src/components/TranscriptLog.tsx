import type { Turn } from '../types';

interface Props {
  turns: Turn[];
}

export default function TranscriptLog({ turns }: Props) {
  if (turns.length === 0) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mt-8 bg-slate-900/40 backdrop-blur border border-white/10 rounded-2xl p-4">
      <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">
        Transcript
      </h3>
      <div className="space-y-3 max-h-80 overflow-y-auto scrollbar-thin pr-2">
        {turns.map((t, idx) => (
          <div key={idx} className="flex gap-3">
            <div
              className="flex-shrink-0 w-2 self-stretch rounded-full"
              style={{ backgroundColor: t.color }}
            />
            <div className="flex-1">
              <div
                className="text-xs font-semibold mb-0.5"
                style={{ color: t.color }}
              >
                {t.name} · Round {t.round}
              </div>
              <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                {t.text}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
