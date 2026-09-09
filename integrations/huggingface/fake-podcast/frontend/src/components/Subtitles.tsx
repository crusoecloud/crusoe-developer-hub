import { useState, useEffect, useRef, useCallback } from 'react';
import type { TranscriptLine } from '../types';

interface Props {
  lines: TranscriptLine[];
  /** Index driven by the audio player — when set, subtitles follow audio */
  audioActiveLine: number | null;
  /** Whether audio is currently playing */
  audioPlaying: boolean;
  isGenerating: boolean;
}

/** Estimate reading time in ms based on word count */
function readingTime(text: string): number {
  const words = text.split(/\s+/).length;
  // ~180 wpm reading speed, minimum 2s, maximum 10s
  return Math.min(10000, Math.max(2000, (words / 180) * 60 * 1000));
}

export default function Subtitles({ lines, audioActiveLine, audioPlaying, isGenerating }: Props) {
  const [textIndex, setTextIndex] = useState(0);
  const [textPlaying, setTextPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The displayed line: audio takes priority when playing, otherwise text-mode index
  const activeIndex = audioPlaying ? (audioActiveLine ?? 0) : (textPlaying ? textIndex : null);
  const activeLine = activeIndex !== null && activeIndex < lines.length ? lines[activeIndex] : null;

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const scheduleNext = useCallback((idx: number) => {
    clearTimer();
    if (idx >= lines.length) {
      setTextPlaying(false);
      setTextIndex(0);
      return;
    }
    const ms = readingTime(lines[idx].text);
    timerRef.current = setTimeout(() => {
      const next = idx + 1;
      setTextIndex(next);
      if (next < lines.length) {
        scheduleNext(next);
      } else {
        setTextPlaying(false);
        setTextIndex(0);
      }
    }, ms);
  }, [lines]);

  // When text-mode starts, kick off the timer
  useEffect(() => {
    if (textPlaying && !audioPlaying && lines.length > 0) {
      scheduleNext(textIndex);
    }
    return clearTimer;
  }, [textPlaying, audioPlaying, lines.length, scheduleNext]);

  // Pause text mode when audio starts playing
  useEffect(() => {
    if (audioPlaying) {
      clearTimer();
    }
  }, [audioPlaying]);

  // Reset when lines change (new episode)
  useEffect(() => {
    setTextIndex(0);
    setTextPlaying(false);
    clearTimer();
  }, [lines.length]);

  const toggleTextMode = () => {
    if (audioPlaying) return; // audio has control
    if (textPlaying) {
      clearTimer();
      setTextPlaying(false);
    } else {
      setTextPlaying(true);
    }
  };

  const skipTo = (direction: 'prev' | 'next') => {
    if (audioPlaying) return;
    clearTimer();
    const newIdx = direction === 'next'
      ? Math.min(textIndex + 1, lines.length - 1)
      : Math.max(textIndex - 1, 0);
    setTextIndex(newIdx);
    if (textPlaying) scheduleNext(newIdx);
  };

  if (lines.length === 0) return null;

  const isActive = audioPlaying || textPlaying;
  const displayIndex = activeIndex ?? 0;

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 overflow-hidden shrink-0">
      {/* Subtitle display area */}
      <div className="relative min-h-[80px] flex items-center justify-center px-6 py-4">
        {isActive && activeLine ? (
          <div className="text-center space-y-1 animate-fade-in" key={displayIndex}>
            <span
              className="text-xs font-semibold uppercase tracking-wider"
              style={{ color: activeLine.color }}
            >
              {activeLine.name}
            </span>
            <p className="text-gray-900 text-base leading-relaxed max-w-2xl">
              {activeLine.text}
            </p>
          </div>
        ) : (
          <p className="text-gray-400 text-sm">
            {isGenerating ? 'Generating...' : 'Press play to start subtitles'}
          </p>
        )}

        {/* Line counter badge */}
        {isActive && (
          <span className="absolute top-2 right-3 text-xs text-gray-400">
            {displayIndex + 1} / {lines.length}
          </span>
        )}

        {/* Mode badge */}
        <span className={`absolute top-2 left-3 text-xs px-2 py-0.5 rounded-full ${
          audioPlaying
            ? 'bg-purple-100 text-purple-600'
            : textPlaying
              ? 'bg-emerald-100 text-emerald-600'
              : 'bg-gray-100 text-gray-400'
        }`}>
          {audioPlaying ? '\u266B Audio' : textPlaying ? '\u25B6 Reading' : 'Paused'}
        </span>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3 px-4 py-2 border-t border-gray-200 bg-white">
        <button
          onClick={() => skipTo('prev')}
          disabled={audioPlaying || displayIndex === 0}
          className="w-8 h-8 rounded-full hover:bg-gray-100 flex items-center justify-center text-gray-500 disabled:opacity-30 transition-colors"
          title="Previous line"
        >
          &#9664;
        </button>

        <button
          onClick={toggleTextMode}
          disabled={audioPlaying || lines.length === 0}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
            textPlaying
              ? 'bg-emerald-600 text-white hover:bg-emerald-500'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300'
          } disabled:opacity-30`}
          title={audioPlaying ? 'Controlled by audio player' : textPlaying ? 'Pause subtitles' : 'Play subtitles (text only)'}
        >
          {textPlaying ? '\u23F8 Pause' : '\u25B6 Read Along'}
        </button>

        <button
          onClick={() => skipTo('next')}
          disabled={audioPlaying || displayIndex >= lines.length - 1}
          className="w-8 h-8 rounded-full hover:bg-gray-100 flex items-center justify-center text-gray-500 disabled:opacity-30 transition-colors"
          title="Next line"
        >
          &#9654;
        </button>
      </div>
    </div>
  );
}
