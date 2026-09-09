import { useEffect, useRef } from 'react';
import type { TranscriptLine } from '../types';

interface Props {
  lines: TranscriptLine[];
  isGenerating: boolean;
  activeLine: number | null;
}

export default function Transcript({ lines, isGenerating, activeLine }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines.length]);

  if (lines.length === 0 && !isGenerating) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center space-y-2">
          <div className="text-5xl">&#127897;</div>
          <p className="text-lg">Pick a topic and hit Generate</p>
          <p className="text-sm">or let us Inspire you</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto space-y-3 pr-2">
      {lines.map((line, i) => (
        <div
          key={i}
          className={`flex gap-3 p-3 rounded-lg transition-all ${
            activeLine === i ? 'line-active bg-gray-100' : 'hover:bg-gray-50'
          }`}
        >
          {/* Avatar */}
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold shrink-0 mt-0.5"
            style={{ backgroundColor: line.color + '30', color: line.color }}
          >
            {line.name.charAt(0)}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold text-sm" style={{ color: line.color }}>
                {line.name}
              </span>
              <span className="text-xs text-gray-400">
                {line.host === 'a' ? 'Qwen3-235B' : 'Nemotron-120B'}
              </span>
            </div>
            <p className={`text-gray-800 leading-relaxed ${
              isGenerating && i === lines.length - 1 ? 'typing-cursor' : ''
            }`}>
              {line.text}
            </p>
          </div>
        </div>
      ))}

      {isGenerating && lines.length > 0 && (
        <div className="flex items-center gap-2 text-gray-500 text-sm pl-12">
          <span className="animate-pulse">&#9679;</span>
          Generating round {lines.length + 1}...
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
