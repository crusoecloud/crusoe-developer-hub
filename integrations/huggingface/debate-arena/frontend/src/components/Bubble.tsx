interface Props {
  text: string;
  color: string;
  streaming: boolean;
  visible: boolean;
  side: 'left' | 'right';
}

/**
 * Speech bubble that appears above a character.
 * Text streams in via the `text` prop. A blinking caret is shown while `streaming`.
 */
export default function Bubble({ text, color, streaming, visible, side }: Props) {
  if (!visible) {
    return <div className="w-full max-w-sm h-28" aria-hidden />;
  }

  const tailAlign = side === 'left' ? 'self-end' : 'self-start';

  return (
    <div
      className={`relative bubble-tail animate-bubble-in rounded-3xl px-5 py-4 shadow-xl text-slate-900 max-w-sm ${tailAlign}`}
      style={{
        backgroundColor: '#fefefe',
        color: '#0f172a',
        borderLeft: `4px solid ${color}`,
      }}
    >
      {/* bubble-tail pseudo-element inherits `currentColor` — wrap so tail matches bubble bg */}
      <div style={{ color: '#fefefe' }}>
        <p className={`text-base leading-snug whitespace-pre-wrap ${streaming ? 'streaming-caret' : ''}`} style={{ color: '#0f172a' }}>
          {text || (streaming ? '' : '\u00A0')}
        </p>
      </div>
    </div>
  );
}
